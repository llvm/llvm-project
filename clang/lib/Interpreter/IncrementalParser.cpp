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
#include "IncrementalAction.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclContextInternals.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/PartialTranslationUnit.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Error.h"

#include <sstream>

#define DEBUG_TYPE "clang-repl"

namespace clang {

class IncrementalPreProcessorTracker : public PPCallbacks {
  std::vector<FileEntryRef> IncludedFiles;
  PartialTranslationUnit::MacroDirectiveInfoQueue MacroDirectives;
  void MacroDirective(const clang::Token &MacroNameTok,
                      const clang::MacroDirective *MD) {
    PartialTranslationUnit::MacroDirectiveInfo MDE(
        MacroNameTok.getIdentifierInfo(), MD);
    MacroDirectives.push_back(MDE);
  }

public:
  explicit IncrementalPreProcessorTracker() {}
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *SuggestedModule,
                          bool ModuleImported,
                          SrcMgr::CharacteristicKind FileType) override {

    if (File) {
      IncludedFiles.push_back(*File);
    }
  }

  /// \name PPCallbacks overrides
  /// Macro support
  void MacroDefined(const clang::Token &MacroNameTok,
                    const clang::MacroDirective *MD) final {
    MacroDirective(MacroNameTok, MD);
  }

  /// \name PPCallbacks overrides
  /// Macro support
  void MacroUndefined(const clang::Token &MacroNameTok,
                      const clang::MacroDefinition & /*MD*/,
                      const clang::MacroDirective *Undef) final {
    if (Undef)
      MacroDirective(MacroNameTok, Undef);
  }

  /// Move out tracked files after Parse completes
  std::vector<FileEntryRef> takeFiles() { return std::move(IncludedFiles); }

  /// Move out tracked macros after Parse completes
  PartialTranslationUnit::MacroDirectiveInfoQueue takeMacros() {
    return std::move(MacroDirectives);
  }

  void reset() {
    IncludedFiles.clear();
    MacroDirectives.clear();
  }
};

// IncrementalParser::IncrementalParser() {}

IncrementalParser::IncrementalParser(CompilerInstance &Instance,
                                     IncrementalAction *Act, llvm::Error &Err,
                                     std::list<PartialTranslationUnit> &PTUs)
    : S(Instance.getSema()), Act(Act), PTUs(PTUs) {
  llvm::ErrorAsOutParameter EAO(&Err);
  Consumer = &S.getASTConsumer();
  P.reset(new Parser(S.getPreprocessor(), S, /*SkipBodies=*/false));

  if (ExternalASTSource *External = S.getASTContext().getExternalSource())
    External->StartTranslationUnit(Consumer);

  P->Initialize();

  // track files that are included when parse to support undo
  auto Tracker = std::make_unique<IncrementalPreProcessorTracker>();
  PreProcessorTracker = Tracker.get();
  S.getPreprocessor().addPPCallbacks(std::move(Tracker));
}

IncrementalParser::~IncrementalParser() { P.reset(); }

llvm::Expected<TranslationUnitDecl *>
IncrementalParser::ParseOrWrapTopLevelDecl() {
  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<Sema> CleanupSema(&S);
  Sema::GlobalEagerInstantiationScope GlobalInstantiations(S, /*Enabled=*/true,
                                                           /*AtEndOfTU=*/true);
  Sema::LocalEagerInstantiationScope LocalInstantiations(S, /*AtEndOfTU=*/true);

  // Add a new PTU.
  ASTContext &C = S.getASTContext();
  C.addTranslationUnitDecl();

  // Skip previous eof due to last incremental input.
  if (P->getCurToken().is(tok::annot_repl_input_end)) {
    P->ConsumeAnyToken();
    // FIXME: Clang does not call ExitScope on finalizing the regular TU, we
    // might want to do that around HandleEndOfTranslationUnit.
    P->ExitScope();
    S.CurContext = nullptr;
    // Start a new PTU.
    P->EnterScope(Scope::DeclScope);
    S.ActOnTranslationUnitScope(P->getCurScope());
  }

  Parser::DeclGroupPtrTy ADecl;
  Sema::ModuleImportState ImportState;
  for (bool AtEOF = P->ParseFirstTopLevelDecl(ADecl, ImportState); !AtEOF;
       AtEOF = P->ParseTopLevelDecl(ADecl, ImportState)) {
    if (ADecl && !Consumer->HandleTopLevelDecl(ADecl.get()))
      return llvm::make_error<llvm::StringError>("Parsing failed. "
                                                 "The consumer rejected a decl",
                                                 std::error_code());
  }

  DiagnosticsEngine &Diags = S.getDiagnostics();
  if (Diags.hasErrorOccurred()) {
    CleanUpTU(C.getTranslationUnitDecl());

    Diags.Reset(/*soft=*/true);
    Diags.getClient()->clear();
    return llvm::make_error<llvm::StringError>("Parsing failed.",
                                               std::error_code());
  }

  // Process any TopLevelDecls generated by #pragma weak.
  for (Decl *D : S.WeakTopLevelDecls()) {
    DeclGroupRef DGR(D);
    Consumer->HandleTopLevelDecl(DGR);
  }

  LocalInstantiations.perform();
  GlobalInstantiations.perform();

  Consumer->HandleTranslationUnit(C);

  return C.getTranslationUnitDecl();
}

llvm::Expected<TranslationUnitDecl *>
IncrementalParser::Parse(llvm::StringRef input) {
  Preprocessor &PP = S.getPreprocessor();
  assert(PP.isIncrementalProcessingEnabled() && "Not in incremental mode!?");

  if (PreProcessorTracker)
    PreProcessorTracker->reset();

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

  SourceManager &SM = S.getSourceManager();

  // FIXME: Create SourceLocation, which will allow clang to order the overload
  // candidates for example
  SourceLocation NewLoc = SM.getLocForStartOfFile(SM.getMainFileID());

  // Create FileID for the current buffer.
  FileID FID;
  // Create FileEntry and FileID for the current buffer.
  FileEntryRef FE = SM.getFileManager().getVirtualFileRef(
      SourceName.str(), InputSize, 0 /* mod time*/);
  SM.overrideFileContents(FE, std::move(MB));

  // Ensure HeaderFileInfo exists before lookup to prevent assertion
  HeaderSearch &HS = PP.getHeaderSearchInfo();
  HS.getFileInfo(FE);

  FID = SM.createFileID(FE, NewLoc, SrcMgr::C_User);

  // NewLoc only used for diags.
  if (PP.EnterSourceFile(FID, /*DirLookup=*/nullptr, NewLoc))
    return llvm::make_error<llvm::StringError>("Parsing failed. "
                                               "Cannot enter source file.",
                                               std::error_code());

  auto PTU = ParseOrWrapTopLevelDecl();
  if (!PTU)
    return PTU.takeError();

  if (PP.getLangOpts().DelayedTemplateParsing) {
    // Microsoft-specific:
    // Late parsed templates can leave unswallowed "macro"-like tokens.
    // They will seriously confuse the Parser when entering the next
    // source file. So lex until we are EOF.
    Token Tok;
    do {
      PP.Lex(Tok);
    } while (Tok.isNot(tok::annot_repl_input_end));
  } else {
    Token AssertTok;
    PP.Lex(AssertTok);
    assert(AssertTok.is(tok::annot_repl_input_end) &&
           "Lexer must be EOF when starting incremental parse!");
  }

  return PTU;
}

void IncrementalParser::CleanUpTU(TranslationUnitDecl *MostRecentTU) {
  if (StoredDeclsMap *Map = MostRecentTU->getPrimaryContext()->getLookupPtr()) {
    for (auto &&[Key, List] : *Map) {
      DeclContextLookupResult R = List.getLookupResult();
      std::vector<NamedDecl *> NamedDeclsToRemove;
      bool RemoveAll = true;
      for (NamedDecl *D : R) {
        if (D->getTranslationUnitDecl() == MostRecentTU)
          NamedDeclsToRemove.push_back(D);
        else
          RemoveAll = false;
      }
      if (LLVM_LIKELY(RemoveAll)) {
        Map->erase(Key);
      } else {
        for (NamedDecl *D : NamedDeclsToRemove)
          List.remove(D);
      }
    }
  }

  ExternCContextDecl *ECCD = S.getASTContext().getExternCContextDecl();
  if (StoredDeclsMap *Map = ECCD->getPrimaryContext()->getLookupPtr()) {
    for (auto &&[Key, List] : *Map) {
      DeclContextLookupResult R = List.getLookupResult();
      llvm::SmallVector<NamedDecl *, 4> NamedDeclsToRemove;
      for (NamedDecl *D : R) {
        // Implicitly generated C decl is not attached to the current TU but
        // lexically attached to the recent TU, so we need to check the lexical
        // context.
        DeclContext *LDC = D->getLexicalDeclContext();
        while (LDC && !isa<TranslationUnitDecl>(LDC))
          LDC = LDC->getLexicalParent();
        TranslationUnitDecl *TopTU = cast_or_null<TranslationUnitDecl>(LDC);
        if (TopTU == MostRecentTU)
          NamedDeclsToRemove.push_back(D);
      }
      for (NamedDecl *D : NamedDeclsToRemove) {
        List.remove(D);
        S.IdResolver.RemoveDecl(D);
      }
    }
  }

  // FIXME: We should de-allocate MostRecentTU
  for (Decl *D : MostRecentTU->decls()) {
    auto *ND = dyn_cast<NamedDecl>(D);
    if (!ND || ND->getDeclName().isEmpty())
      continue;
    // Check if we need to clean up the IdResolver chain.
    if (ND->getDeclName().getFETokenInfo() && !D->getLangOpts().ObjC &&
        !D->getLangOpts().CPlusPlus)
      S.IdResolver.RemoveDecl(ND);
  }
}

void IncrementalParser::CleanUpPTU(
    const PartialTranslationUnit &MostRecentPTU) {
  CleanUpTU(MostRecentPTU.TUPart);

  auto &PP = P->getPreprocessor();
  auto &HS = PP.getHeaderSearchInfo();
  auto &SM = PP.getSourceManager();

  auto &MacroDirectives = PTUs.back().MacroDirectiveInfos;
  bool Successful = true;
  if (!MacroDirectives.empty()) {
    for (auto MI = MacroDirectives.rbegin(), ME = MacroDirectives.rend();
         MI != ME; ++MI) {
      // Get rid of the macro definition
      auto UnloadMacro =
          [&PP](PartialTranslationUnit::MacroDirectiveInfo MacroD) {
            const MacroDirective *MD = MacroD.MD;
            // Undef the definition
            const MacroInfo *MI = MD->getMacroInfo();

            // If the macro is not defined, this is a noop undef, just return.
            if (!MI)
              return false;

            // Remove the pair from the macros
            PP.removeMacro(MacroD.II, const_cast<MacroDirective *>(MacroD.MD));

            return true;
          };
      Successful = UnloadMacro(*MI) && Successful;
    }
  }
  if (!PTUs.back().IncludedFiles.empty()) {
    for (FileEntryRef FE : PTUs.back().IncludedFiles) {
      HeaderFileInfo &HFI = HS.getFileInfo(FE);
      HFI.IsLocallyIncluded = false;
      HFI.isPragmaOnce = false;
      HFI.LazyControllingMacro = LazyIdentifierInfoPtr();
      HFI.IsValid = false;

      SM.invalidateCache(SM.translateFile(FE));
    }
  }
}

PartialTranslationUnit &
IncrementalParser::RegisterPTU(TranslationUnitDecl *TU,
                               std::unique_ptr<llvm::Module> M /*={}*/) {
  PTUs.emplace_back(PartialTranslationUnit());
  PartialTranslationUnit &LastPTU = PTUs.back();
  LastPTU.TUPart = TU;

  if (!M)
    M = Act->GenModule();

  assert((!Act->getCodeGen() || M) && "Must have a llvm::Module at this point");

  LastPTU.TheModule = std::move(M);
  LLVM_DEBUG(llvm::dbgs() << "compile-ptu " << PTUs.size() - 1
                          << ": [TU=" << LastPTU.TUPart);
  if (LastPTU.TheModule)
    LLVM_DEBUG(llvm::dbgs() << ", M=" << LastPTU.TheModule.get() << " ("
                            << LastPTU.TheModule->getName() << ")");
  LLVM_DEBUG(llvm::dbgs() << "]\n");

  if (PreProcessorTracker) {
    LastPTU.IncludedFiles = PreProcessorTracker->takeFiles();
    LastPTU.MacroDirectiveInfos = PreProcessorTracker->takeMacros();
  }
  return LastPTU;
}
} // end namespace clang
