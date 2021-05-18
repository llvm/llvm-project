//===--------- IncrementalParser.cpp - Incremental Compilation  -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the class which performs incremental code compilation.
//
//===----------------------------------------------------------------------===//

#include "IncrementalParser.h"

#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"

#include "llvm/Option/ArgList.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Timer.h"

#include <sstream>

namespace clang {

/// A custom action enabling the incremental processing functionality.
///
/// The usual \p FrontendAction expects one call to ExecuteAction and once it
/// sees a call to \p EndSourceFile it deletes some of the important objects
/// such as \p Preprocessor and \p Sema assuming no further input will come.
///
/// \p IncrementalAction ensures it keep its underlying action's objects alive
/// as long as the \p IncrementalParser needs them.
///
class IncrementalAction : public WrapperFrontendAction {
private:
  bool IsTerminating = false;

public:
  IncrementalAction(CompilerInstance &CI, llvm::LLVMContext &LLVMCtx,
                    llvm::Error &Err)
      : WrapperFrontendAction([&]() {
          llvm::ErrorAsOutParameter EAO(&Err);
          std::unique_ptr<FrontendAction> Act;
          switch (CI.getFrontendOpts().ProgramAction) {
          default:
            Err = llvm::createStringError(
                std::errc::state_not_recoverable,
                "Driver initialization failed. "
                "Incremental mode for action %d is not supported",
                CI.getFrontendOpts().ProgramAction);
            return Act;
          case frontend::ASTDump:
            LLVM_FALLTHROUGH;
          case frontend::ASTPrint:
            LLVM_FALLTHROUGH;
          case frontend::ParseSyntaxOnly:
            Act = CreateFrontendAction(CI);
            break;
          case frontend::EmitAssembly:
            LLVM_FALLTHROUGH;
          case frontend::EmitObj:
            LLVM_FALLTHROUGH;
          case frontend::EmitLLVMOnly:
            Act.reset(new EmitLLVMOnlyAction(&LLVMCtx));
            break;
          }
          return Act;
        }()) {}
  FrontendAction *getWrapped() const { return WrappedAction.get(); }
  void ExecuteAction() override {
    CompilerInstance &CI = getCompilerInstance();
    assert(CI.hasPreprocessor() && "No PP!");

    // FIXME: Move the truncation aspect of this into Sema, we delayed this till
    // here so the source manager would be initialized.
    if (hasCodeCompletionSupport() &&
        !CI.getFrontendOpts().CodeCompletionAt.FileName.empty())
      CI.createCodeCompletionConsumer();

    // Use a code completion consumer?
    CodeCompleteConsumer *CompletionConsumer = nullptr;
    if (CI.hasCodeCompletionConsumer())
      CompletionConsumer = &CI.getCodeCompletionConsumer();

    Preprocessor &PP = CI.getPreprocessor();
    PP.enableIncrementalProcessing();
    PP.EnterMainSourceFile();

    if (!CI.hasSema())
      CI.createSema(getTranslationUnitKind(), CompletionConsumer);
  }

  // Do not terminate after processing the input. This allows us to keep various
  // clang objects alive and to incrementally grow the current TU.
  void EndSourceFile() override {
    // The WrappedAction can be nullptr if we issued an error in the ctor.
    if (IsTerminating && getWrapped())
      WrapperFrontendAction::EndSourceFile();
  }

  void FinalizeAction() {
    assert(!IsTerminating && "Already finalized!");
    IsTerminating = true;
    EndSourceFile();
  }
};

IncrementalParser::IncrementalParser(std::unique_ptr<CompilerInstance> Instance,
                                     llvm::LLVMContext &LLVMCtx,
                                     llvm::Error &Err)
    : CI(std::move(Instance)) {
  llvm::ErrorAsOutParameter EAO(&Err);
  Act = std::make_unique<IncrementalAction>(*CI, LLVMCtx, Err);
  if (Err)
    return;
  CI->ExecuteAction(*Act);
  Consumer = &CI->getASTConsumer();
  P.reset(
      new Parser(CI->getPreprocessor(), CI->getSema(), /*SkipBodies=*/false));
  P->Initialize();
}

IncrementalParser::~IncrementalParser() { Act->FinalizeAction(); }

llvm::Expected<Transaction &> IncrementalParser::ParseOrWrapTopLevelDecl() {
  DiagnosticsEngine &Diags = getCI()->getDiagnostics();

  if (Diags.hasErrorOccurred())
    llvm::report_fatal_error("Previous input had errors, "
                             "recovery not yet supported",
                             /*GenCrashDiag=*/false);

  // Recover resources if we crash before exiting this method.
  Sema &S = CI->getSema();
  llvm::CrashRecoveryContextCleanupRegistrar<Sema> CleanupSema(&S);
  Sema::GlobalEagerInstantiationScope GlobalInstantiations(S, /*Enabled=*/true);
  Sema::LocalEagerInstantiationScope LocalInstantiations(S);

  // Skip previous eof due to last incremental input.
  if (P->getCurToken().is(tok::eof))
    P->ConsumeToken();

  Transactions.emplace_back(Transaction());
  Transaction &LastTransaction = Transactions.back();

  Parser::DeclGroupPtrTy ADecl;
  for (bool AtEOF = P->ParseFirstTopLevelDecl(ADecl); !AtEOF;
       AtEOF = P->ParseTopLevelDecl(ADecl)) {
    // If we got a null return and something *was* parsed, ignore it.  This
    // is due to a top-level semicolon, an action override, or a parse error
    // skipping something.
    if (ADecl && !Consumer->HandleTopLevelDecl(ADecl.get()))
      return llvm::make_error<llvm::StringError>("Parsing failed. "
                                                 "The consumer rejected a decl",
                                                 std::error_code());
    LastTransaction.Decls.push_back(ADecl.get());
  }

  // Process any TopLevelDecls generated by #pragma weak.
  for (Decl *D : S.WeakTopLevelDecls()) {
    DeclGroupRef DGR(D);
    LastTransaction.Decls.push_back(DGR);
    Consumer->HandleTopLevelDecl(DGR);
  }

  LocalInstantiations.perform();
  GlobalInstantiations.perform();

  Consumer->HandleTranslationUnit(S.getASTContext());

  if (Diags.hasErrorOccurred())
    return llvm::make_error<llvm::StringError>("Parsing failed.",
                                               std::error_code());

  return LastTransaction;
}

static CodeGenerator *getCodeGen(FrontendAction *Act) {
  IncrementalAction *IncrAct = static_cast<IncrementalAction *>(Act);
  FrontendAction *WrappedAct = IncrAct->getWrapped();
  if (!WrappedAct->hasIRSupport())
    return nullptr;
  return static_cast<CodeGenAction *>(WrappedAct)->getCodeGenerator();
}

llvm::Expected<Transaction &> IncrementalParser::Parse(llvm::StringRef input) {
  Preprocessor &PP = CI->getPreprocessor();
  assert(PP.isIncrementalProcessingEnabled() && "Not in incremental mode!?");

  std::ostringstream SourceName;
  SourceName << "input_line_" << InputCount++;

  // Create an uninitialized memory buffer, copy code in and append "\n"
  size_t InputSize = input.size(); // don't include trailing 0
  // MemBuffer size should *not* include terminating zero
  std::unique_ptr<llvm::MemoryBuffer> MB(
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(InputSize + 1,
                                                        SourceName.str()));
  char *MBStart = const_cast<char *>(MB->getBufferStart());
  memcpy(MBStart, input.data(), InputSize);
  MBStart[InputSize] = '\n';

  SourceManager &SM = CI->getSourceManager();

  // FIXME: Create SourceLocation, which will allow clang to order the overload
  // candidates for example
  SourceLocation NewLoc = SM.getLocForStartOfFile(SM.getMainFileID());

  // Create FileID for the current buffer.
  FileID FID = SM.createFileID(std::move(MB), SrcMgr::C_User, /*LoadedID=*/0,
                               /*LoadedOffset=*/0, NewLoc);

  // NewLoc only used for diags.
  if (PP.EnterSourceFile(FID, /*DirLookup=*/0, NewLoc))
    return llvm::make_error<llvm::StringError>("Parsing failed. "
                                               "Cannot enter source file.",
                                               std::error_code());

  auto ErrOrTransaction = ParseOrWrapTopLevelDecl();
  if (auto Err = ErrOrTransaction.takeError())
    return std::move(Err);

  if (PP.getLangOpts().DelayedTemplateParsing) {
    // Microsoft-specific:
    // Late parsed templates can leave unswallowed "macro"-like tokens.
    // They will seriously confuse the Parser when entering the next
    // source file. So lex until we are EOF.
    Token Tok;
    do {
      PP.Lex(Tok);
    } while (Tok.isNot(tok::eof));
  }

  Token AssertTok;
  PP.Lex(AssertTok);
  assert(AssertTok.is(tok::eof) &&
         "Lexer must be EOF when starting incremental parse!");

  if (CodeGenerator *CG = getCodeGen(Act.get())) {
    std::unique_ptr<llvm::Module> M(CG->ReleaseModule());
    CG->StartModule("incr_module_" + std::to_string(Transactions.size()),
                    M->getContext());

    ErrOrTransaction->TheModule = std::move(M);
  }

  return ErrOrTransaction;
}
} // end namespace clang
