//===--- ClangdUnit.cpp -----------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "ClangdUnit.h"

#include "Logger.h"
#include "Trace.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/Utils.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/Sema.h"
#include "clang/Serialization/ASTWriter.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <chrono>

using namespace clang::clangd;
using namespace clang;

namespace {

class DeclTrackingASTConsumer : public ASTConsumer {
public:
  DeclTrackingASTConsumer(std::vector<const Decl *> &TopLevelDecls)
      : TopLevelDecls(TopLevelDecls) {}

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    for (const Decl *D : DG) {
      // ObjCMethodDecl are not actually top-level decls.
      if (isa<ObjCMethodDecl>(D))
        continue;

      TopLevelDecls.push_back(D);
    }
    return true;
  }

private:
  std::vector<const Decl *> &TopLevelDecls;
};

class ClangdFrontendAction : public SyntaxOnlyAction {
public:
  std::vector<const Decl *> takeTopLevelDecls() {
    return std::move(TopLevelDecls);
  }

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return llvm::make_unique<DeclTrackingASTConsumer>(/*ref*/ TopLevelDecls);
  }

private:
  std::vector<const Decl *> TopLevelDecls;
};

class CppFilePreambleCallbacks : public PreambleCallbacks {
public:
  std::vector<serialization::DeclID> takeTopLevelDeclIDs() {
    return std::move(TopLevelDeclIDs);
  }

  void AfterPCHEmitted(ASTWriter &Writer) override {
    TopLevelDeclIDs.reserve(TopLevelDecls.size());
    for (Decl *D : TopLevelDecls) {
      // Invalid top-level decls may not have been serialized.
      if (D->isInvalidDecl())
        continue;
      TopLevelDeclIDs.push_back(Writer.getDeclID(D));
    }
  }

  void HandleTopLevelDecl(DeclGroupRef DG) override {
    for (Decl *D : DG) {
      if (isa<ObjCMethodDecl>(D))
        continue;
      TopLevelDecls.push_back(D);
    }
  }

private:
  std::vector<Decl *> TopLevelDecls;
  std::vector<serialization::DeclID> TopLevelDeclIDs;
};

/// Convert from clang diagnostic level to LSP severity.
static int getSeverity(DiagnosticsEngine::Level L) {
  switch (L) {
  case DiagnosticsEngine::Remark:
    return 4;
  case DiagnosticsEngine::Note:
    return 3;
  case DiagnosticsEngine::Warning:
    return 2;
  case DiagnosticsEngine::Fatal:
  case DiagnosticsEngine::Error:
    return 1;
  case DiagnosticsEngine::Ignored:
    return 0;
  }
  llvm_unreachable("Unknown diagnostic level!");
}

/// Get the optional chunk as a string. This function is possibly recursive.
///
/// The parameter info for each parameter is appended to the Parameters.
std::string
getOptionalParameters(const CodeCompletionString &CCS,
                      std::vector<ParameterInformation> &Parameters) {
  std::string Result;
  for (const auto &Chunk : CCS) {
    switch (Chunk.Kind) {
    case CodeCompletionString::CK_Optional:
      assert(Chunk.Optional &&
             "Expected the optional code completion string to be non-null.");
      Result += getOptionalParameters(*Chunk.Optional, Parameters);
      break;
    case CodeCompletionString::CK_VerticalSpace:
      break;
    case CodeCompletionString::CK_Placeholder:
      // A string that acts as a placeholder for, e.g., a function call
      // argument.
      // Intentional fallthrough here.
    case CodeCompletionString::CK_CurrentParameter: {
      // A piece of text that describes the parameter that corresponds to
      // the code-completion location within a function call, message send,
      // macro invocation, etc.
      Result += Chunk.Text;
      ParameterInformation Info;
      Info.label = Chunk.Text;
      Parameters.push_back(std::move(Info));
      break;
    }
    default:
      Result += Chunk.Text;
      break;
    }
  }
  return Result;
}

llvm::Optional<DiagWithFixIts> toClangdDiag(const StoredDiagnostic &D) {
  auto Location = D.getLocation();
  if (!Location.isValid() || !Location.getManager().isInMainFile(Location))
    return llvm::None;

  Position P;
  P.line = Location.getSpellingLineNumber() - 1;
  P.character = Location.getSpellingColumnNumber();
  Range R = {P, P};
  clangd::Diagnostic Diag = {R, getSeverity(D.getLevel()), D.getMessage()};

  llvm::SmallVector<tooling::Replacement, 1> FixItsForDiagnostic;
  for (const FixItHint &Fix : D.getFixIts()) {
    FixItsForDiagnostic.push_back(clang::tooling::Replacement(
        Location.getManager(), Fix.RemoveRange, Fix.CodeToInsert));
  }
  return DiagWithFixIts{Diag, std::move(FixItsForDiagnostic)};
}

class StoreDiagsConsumer : public DiagnosticConsumer {
public:
  StoreDiagsConsumer(std::vector<DiagWithFixIts> &Output) : Output(Output) {}

  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const clang::Diagnostic &Info) override {
    DiagnosticConsumer::HandleDiagnostic(DiagLevel, Info);

    if (auto convertedDiag = toClangdDiag(StoredDiagnostic(DiagLevel, Info)))
      Output.push_back(std::move(*convertedDiag));
  }

private:
  std::vector<DiagWithFixIts> &Output;
};

class EmptyDiagsConsumer : public DiagnosticConsumer {
public:
  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const clang::Diagnostic &Info) override {}
};

std::unique_ptr<CompilerInvocation>
createCompilerInvocation(ArrayRef<const char *> ArgList,
                         IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
                         IntrusiveRefCntPtr<vfs::FileSystem> VFS) {
  auto CI = createInvocationFromCommandLine(ArgList, std::move(Diags),
                                            std::move(VFS));
  // We rely on CompilerInstance to manage the resource (i.e. free them on
  // EndSourceFile), but that won't happen if DisableFree is set to true.
  // Since createInvocationFromCommandLine sets it to true, we have to override
  // it.
  CI->getFrontendOpts().DisableFree = false;
  return CI;
}

/// Creates a CompilerInstance from \p CI, with main buffer overriden to \p
/// Buffer and arguments to read the PCH from \p Preamble, if \p Preamble is not
/// null. Note that vfs::FileSystem inside returned instance may differ from \p
/// VFS if additional file remapping were set in command-line arguments.
/// On some errors, returns null. When non-null value is returned, it's expected
/// to be consumed by the FrontendAction as it will have a pointer to the \p
/// Buffer that will only be deleted if BeginSourceFile is called.
std::unique_ptr<CompilerInstance>
prepareCompilerInstance(std::unique_ptr<clang::CompilerInvocation> CI,
                        const PrecompiledPreamble *Preamble,
                        std::unique_ptr<llvm::MemoryBuffer> Buffer,
                        std::shared_ptr<PCHContainerOperations> PCHs,
                        IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                        DiagnosticConsumer &DiagsClient) {
  assert(VFS && "VFS is null");
  assert(!CI->getPreprocessorOpts().RetainRemappedFileBuffers &&
         "Setting RetainRemappedFileBuffers to true will cause a memory leak "
         "of ContentsBuffer");

  // NOTE: we use Buffer.get() when adding remapped files, so we have to make
  // sure it will be released if no error is emitted.
  if (Preamble) {
    Preamble->AddImplicitPreamble(*CI, Buffer.get());
  } else {
    CI->getPreprocessorOpts().addRemappedFile(
        CI->getFrontendOpts().Inputs[0].getFile(), Buffer.get());
  }

  auto Clang = llvm::make_unique<CompilerInstance>(PCHs);
  Clang->setInvocation(std::move(CI));
  Clang->createDiagnostics(&DiagsClient, false);

  if (auto VFSWithRemapping = createVFSFromCompilerInvocation(
          Clang->getInvocation(), Clang->getDiagnostics(), VFS))
    VFS = VFSWithRemapping;
  Clang->setVirtualFileSystem(VFS);

  Clang->setTarget(TargetInfo::CreateTargetInfo(
      Clang->getDiagnostics(), Clang->getInvocation().TargetOpts));
  if (!Clang->hasTarget())
    return nullptr;

  // RemappedFileBuffers will handle the lifetime of the Buffer pointer,
  // release it.
  Buffer.release();
  return Clang;
}

template <class T> bool futureIsReady(std::shared_future<T> const &Future) {
  return Future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

} // namespace

namespace {

CompletionItemKind getKindOfDecl(CXCursorKind CursorKind) {
  switch (CursorKind) {
  case CXCursor_MacroInstantiation:
  case CXCursor_MacroDefinition:
    return CompletionItemKind::Text;
  case CXCursor_CXXMethod:
    return CompletionItemKind::Method;
  case CXCursor_FunctionDecl:
  case CXCursor_FunctionTemplate:
    return CompletionItemKind::Function;
  case CXCursor_Constructor:
  case CXCursor_Destructor:
    return CompletionItemKind::Constructor;
  case CXCursor_FieldDecl:
    return CompletionItemKind::Field;
  case CXCursor_VarDecl:
  case CXCursor_ParmDecl:
    return CompletionItemKind::Variable;
  case CXCursor_ClassDecl:
  case CXCursor_StructDecl:
  case CXCursor_UnionDecl:
  case CXCursor_ClassTemplate:
  case CXCursor_ClassTemplatePartialSpecialization:
    return CompletionItemKind::Class;
  case CXCursor_Namespace:
  case CXCursor_NamespaceAlias:
  case CXCursor_NamespaceRef:
    return CompletionItemKind::Module;
  case CXCursor_EnumConstantDecl:
    return CompletionItemKind::Value;
  case CXCursor_EnumDecl:
    return CompletionItemKind::Enum;
  case CXCursor_TypeAliasDecl:
  case CXCursor_TypeAliasTemplateDecl:
  case CXCursor_TypedefDecl:
  case CXCursor_MemberRef:
  case CXCursor_TypeRef:
    return CompletionItemKind::Reference;
  default:
    return CompletionItemKind::Missing;
  }
}

CompletionItemKind getKind(CodeCompletionResult::ResultKind ResKind,
                           CXCursorKind CursorKind) {
  switch (ResKind) {
  case CodeCompletionResult::RK_Declaration:
    return getKindOfDecl(CursorKind);
  case CodeCompletionResult::RK_Keyword:
    return CompletionItemKind::Keyword;
  case CodeCompletionResult::RK_Macro:
    return CompletionItemKind::Text; // unfortunately, there's no 'Macro'
                                     // completion items in LSP.
  case CodeCompletionResult::RK_Pattern:
    return CompletionItemKind::Snippet;
  }
  llvm_unreachable("Unhandled CodeCompletionResult::ResultKind.");
}

std::string escapeSnippet(const llvm::StringRef Text) {
  std::string Result;
  Result.reserve(Text.size()); // Assume '$', '}' and '\\' are rare.
  for (const auto Character : Text) {
    if (Character == '$' || Character == '}' || Character == '\\')
      Result.push_back('\\');
    Result.push_back(Character);
  }
  return Result;
}

std::string getDocumentation(const CodeCompletionString &CCS) {
  // Things like __attribute__((nonnull(1,3))) and [[noreturn]]. Present this
  // information in the documentation field.
  std::string Result;
  const unsigned AnnotationCount = CCS.getAnnotationCount();
  if (AnnotationCount > 0) {
    Result += "Annotation";
    if (AnnotationCount == 1) {
      Result += ": ";
    } else /* AnnotationCount > 1 */ {
      Result += "s: ";
    }
    for (unsigned I = 0; I < AnnotationCount; ++I) {
      Result += CCS.getAnnotation(I);
      Result.push_back(I == AnnotationCount - 1 ? '\n' : ' ');
    }
  }
  // Add brief documentation (if there is any).
  if (CCS.getBriefComment() != nullptr) {
    if (!Result.empty()) {
      // This means we previously added annotations. Add an extra newline
      // character to make the annotations stand out.
      Result.push_back('\n');
    }
    Result += CCS.getBriefComment();
  }
  return Result;
}

class CompletionItemsCollector : public CodeCompleteConsumer {
public:
  CompletionItemsCollector(const clang::CodeCompleteOptions &CodeCompleteOpts,
                           std::vector<CompletionItem> &Items)
      : CodeCompleteConsumer(CodeCompleteOpts, /*OutputIsBinary=*/false),
        Items(Items),
        Allocator(std::make_shared<clang::GlobalCodeCompletionAllocator>()),
        CCTUInfo(Allocator) {}

  void ProcessCodeCompleteResults(Sema &S, CodeCompletionContext Context,
                                  CodeCompletionResult *Results,
                                  unsigned NumResults) override final {
    Items.reserve(NumResults);
    for (unsigned I = 0; I < NumResults; ++I) {
      auto &Result = Results[I];
      const auto *CCS = Result.CreateCodeCompletionString(
          S, Context, *Allocator, CCTUInfo,
          CodeCompleteOpts.IncludeBriefComments);
      assert(CCS && "Expected the CodeCompletionString to be non-null");
      Items.push_back(ProcessCodeCompleteResult(Result, *CCS));
    }
    std::sort(Items.begin(), Items.end());
  }

  GlobalCodeCompletionAllocator &getAllocator() override { return *Allocator; }

  CodeCompletionTUInfo &getCodeCompletionTUInfo() override { return CCTUInfo; }

private:
  CompletionItem
  ProcessCodeCompleteResult(const CodeCompletionResult &Result,
                            const CodeCompletionString &CCS) const {

    // Adjust this to InsertTextFormat::Snippet iff we encounter a
    // CK_Placeholder chunk in SnippetCompletionItemsCollector.
    CompletionItem Item;
    Item.insertTextFormat = InsertTextFormat::PlainText;

    Item.documentation = getDocumentation(CCS);

    // Fill in the label, detail, insertText and filterText fields of the
    // CompletionItem.
    ProcessChunks(CCS, Item);

    // Fill in the kind field of the CompletionItem.
    Item.kind = getKind(Result.Kind, Result.CursorKind);

    FillSortText(CCS, Item);

    return Item;
  }

  virtual void ProcessChunks(const CodeCompletionString &CCS,
                             CompletionItem &Item) const = 0;

  static int GetSortPriority(const CodeCompletionString &CCS) {
    int Score = CCS.getPriority();
    // Fill in the sortText of the CompletionItem.
    assert(Score <= 99999 && "Expecting code completion result "
                             "priority to have at most 5-digits");

    const int Penalty = 100000;
    switch (static_cast<CXAvailabilityKind>(CCS.getAvailability())) {
    case CXAvailability_Available:
      // No penalty.
      break;
    case CXAvailability_Deprecated:
      Score += Penalty;
      break;
    case CXAvailability_NotAccessible:
      Score += 2 * Penalty;
      break;
    case CXAvailability_NotAvailable:
      Score += 3 * Penalty;
      break;
    }

    return Score;
  }

  static void FillSortText(const CodeCompletionString &CCS,
                           CompletionItem &Item) {
    int Priority = GetSortPriority(CCS);
    // Fill in the sortText of the CompletionItem.
    assert(Priority <= 999999 &&
           "Expecting sort priority to have at most 6-digits");
    llvm::raw_string_ostream(Item.sortText)
        << llvm::format("%06d%s", Priority, Item.filterText.c_str());
  }

  std::vector<CompletionItem> &Items;
  std::shared_ptr<clang::GlobalCodeCompletionAllocator> Allocator;
  CodeCompletionTUInfo CCTUInfo;

}; // CompletionItemsCollector

bool isInformativeQualifierChunk(CodeCompletionString::Chunk const &Chunk) {
  return Chunk.Kind == CodeCompletionString::CK_Informative &&
         StringRef(Chunk.Text).endswith("::");
}

class PlainTextCompletionItemsCollector final
    : public CompletionItemsCollector {

public:
  PlainTextCompletionItemsCollector(
      const clang::CodeCompleteOptions &CodeCompleteOpts,
      std::vector<CompletionItem> &Items)
      : CompletionItemsCollector(CodeCompleteOpts, Items) {}

private:
  void ProcessChunks(const CodeCompletionString &CCS,
                     CompletionItem &Item) const override {
    for (const auto &Chunk : CCS) {
      // Informative qualifier chunks only clutter completion results, skip
      // them.
      if (isInformativeQualifierChunk(Chunk))
        continue;

      switch (Chunk.Kind) {
      case CodeCompletionString::CK_TypedText:
        // There's always exactly one CK_TypedText chunk.
        Item.insertText = Item.filterText = Chunk.Text;
        Item.label += Chunk.Text;
        break;
      case CodeCompletionString::CK_ResultType:
        assert(Item.detail.empty() && "Unexpected extraneous CK_ResultType");
        Item.detail = Chunk.Text;
        break;
      case CodeCompletionString::CK_Optional:
        break;
      default:
        Item.label += Chunk.Text;
        break;
      }
    }
  }
}; // PlainTextCompletionItemsCollector

class SnippetCompletionItemsCollector final : public CompletionItemsCollector {

public:
  SnippetCompletionItemsCollector(
      const clang::CodeCompleteOptions &CodeCompleteOpts,
      std::vector<CompletionItem> &Items)
      : CompletionItemsCollector(CodeCompleteOpts, Items) {}

private:
  void ProcessChunks(const CodeCompletionString &CCS,
                     CompletionItem &Item) const override {
    unsigned ArgCount = 0;
    for (const auto &Chunk : CCS) {
      // Informative qualifier chunks only clutter completion results, skip
      // them.
      if (isInformativeQualifierChunk(Chunk))
        continue;

      switch (Chunk.Kind) {
      case CodeCompletionString::CK_TypedText:
        // The piece of text that the user is expected to type to match
        // the code-completion string, typically a keyword or the name of
        // a declarator or macro.
        Item.filterText = Chunk.Text;
        LLVM_FALLTHROUGH;
      case CodeCompletionString::CK_Text:
        // A piece of text that should be placed in the buffer,
        // e.g., parentheses or a comma in a function call.
        Item.label += Chunk.Text;
        Item.insertText += Chunk.Text;
        break;
      case CodeCompletionString::CK_Optional:
        // A code completion string that is entirely optional.
        // For example, an optional code completion string that
        // describes the default arguments in a function call.

        // FIXME: Maybe add an option to allow presenting the optional chunks?
        break;
      case CodeCompletionString::CK_Placeholder:
        // A string that acts as a placeholder for, e.g., a function call
        // argument.
        ++ArgCount;
        Item.insertText += "${" + std::to_string(ArgCount) + ':' +
                           escapeSnippet(Chunk.Text) + '}';
        Item.label += Chunk.Text;
        Item.insertTextFormat = InsertTextFormat::Snippet;
        break;
      case CodeCompletionString::CK_Informative:
        // A piece of text that describes something about the result
        // but should not be inserted into the buffer.
        // For example, the word "const" for a const method, or the name of
        // the base class for methods that are part of the base class.
        Item.label += Chunk.Text;
        // Don't put the informative chunks in the insertText.
        break;
      case CodeCompletionString::CK_ResultType:
        // A piece of text that describes the type of an entity or,
        // for functions and methods, the return type.
        assert(Item.detail.empty() && "Unexpected extraneous CK_ResultType");
        Item.detail = Chunk.Text;
        break;
      case CodeCompletionString::CK_CurrentParameter:
        // A piece of text that describes the parameter that corresponds to
        // the code-completion location within a function call, message send,
        // macro invocation, etc.
        //
        // This should never be present while collecting completion items,
        // only while collecting overload candidates.
        llvm_unreachable("Unexpected CK_CurrentParameter while collecting "
                         "CompletionItems");
        break;
      case CodeCompletionString::CK_LeftParen:
        // A left parenthesis ('(').
      case CodeCompletionString::CK_RightParen:
        // A right parenthesis (')').
      case CodeCompletionString::CK_LeftBracket:
        // A left bracket ('[').
      case CodeCompletionString::CK_RightBracket:
        // A right bracket (']').
      case CodeCompletionString::CK_LeftBrace:
        // A left brace ('{').
      case CodeCompletionString::CK_RightBrace:
        // A right brace ('}').
      case CodeCompletionString::CK_LeftAngle:
        // A left angle bracket ('<').
      case CodeCompletionString::CK_RightAngle:
        // A right angle bracket ('>').
      case CodeCompletionString::CK_Comma:
        // A comma separator (',').
      case CodeCompletionString::CK_Colon:
        // A colon (':').
      case CodeCompletionString::CK_SemiColon:
        // A semicolon (';').
      case CodeCompletionString::CK_Equal:
        // An '=' sign.
      case CodeCompletionString::CK_HorizontalSpace:
        // Horizontal whitespace (' ').
        Item.insertText += Chunk.Text;
        Item.label += Chunk.Text;
        break;
      case CodeCompletionString::CK_VerticalSpace:
        // Vertical whitespace ('\n' or '\r\n', depending on the
        // platform).
        Item.insertText += Chunk.Text;
        // Don't even add a space to the label.
        break;
      }
    }
  }
}; // SnippetCompletionItemsCollector

class SignatureHelpCollector final : public CodeCompleteConsumer {

public:
  SignatureHelpCollector(const clang::CodeCompleteOptions &CodeCompleteOpts,
                         SignatureHelp &SigHelp)
      : CodeCompleteConsumer(CodeCompleteOpts, /*OutputIsBinary=*/false),
        SigHelp(SigHelp),
        Allocator(std::make_shared<clang::GlobalCodeCompletionAllocator>()),
        CCTUInfo(Allocator) {}

  void ProcessOverloadCandidates(Sema &S, unsigned CurrentArg,
                                 OverloadCandidate *Candidates,
                                 unsigned NumCandidates) override {
    SigHelp.signatures.reserve(NumCandidates);
    // FIXME(rwols): How can we determine the "active overload candidate"?
    // Right now the overloaded candidates seem to be provided in a "best fit"
    // order, so I'm not too worried about this.
    SigHelp.activeSignature = 0;
    assert(CurrentArg <= (unsigned)std::numeric_limits<int>::max() &&
           "too many arguments");
    SigHelp.activeParameter = static_cast<int>(CurrentArg);
    for (unsigned I = 0; I < NumCandidates; ++I) {
      const auto &Candidate = Candidates[I];
      const auto *CCS = Candidate.CreateSignatureString(
          CurrentArg, S, *Allocator, CCTUInfo, true);
      assert(CCS && "Expected the CodeCompletionString to be non-null");
      SigHelp.signatures.push_back(ProcessOverloadCandidate(Candidate, *CCS));
    }
  }

  GlobalCodeCompletionAllocator &getAllocator() override { return *Allocator; }

  CodeCompletionTUInfo &getCodeCompletionTUInfo() override { return CCTUInfo; }

private:
  SignatureInformation
  ProcessOverloadCandidate(const OverloadCandidate &Candidate,
                           const CodeCompletionString &CCS) const {
    SignatureInformation Result;
    const char *ReturnType = nullptr;

    Result.documentation = getDocumentation(CCS);

    for (const auto &Chunk : CCS) {
      switch (Chunk.Kind) {
      case CodeCompletionString::CK_ResultType:
        // A piece of text that describes the type of an entity or,
        // for functions and methods, the return type.
        assert(!ReturnType && "Unexpected CK_ResultType");
        ReturnType = Chunk.Text;
        break;
      case CodeCompletionString::CK_Placeholder:
        // A string that acts as a placeholder for, e.g., a function call
        // argument.
        // Intentional fallthrough here.
      case CodeCompletionString::CK_CurrentParameter: {
        // A piece of text that describes the parameter that corresponds to
        // the code-completion location within a function call, message send,
        // macro invocation, etc.
        Result.label += Chunk.Text;
        ParameterInformation Info;
        Info.label = Chunk.Text;
        Result.parameters.push_back(std::move(Info));
        break;
      }
      case CodeCompletionString::CK_Optional: {
        // The rest of the parameters are defaulted/optional.
        assert(Chunk.Optional &&
               "Expected the optional code completion string to be non-null.");
        Result.label +=
            getOptionalParameters(*Chunk.Optional, Result.parameters);
        break;
      }
      case CodeCompletionString::CK_VerticalSpace:
        break;
      default:
        Result.label += Chunk.Text;
        break;
      }
    }
    if (ReturnType) {
      Result.label += " -> ";
      Result.label += ReturnType;
    }
    return Result;
  }

  SignatureHelp &SigHelp;
  std::shared_ptr<clang::GlobalCodeCompletionAllocator> Allocator;
  CodeCompletionTUInfo CCTUInfo;

}; // SignatureHelpCollector

bool invokeCodeComplete(std::unique_ptr<CodeCompleteConsumer> Consumer,
                        const clang::CodeCompleteOptions &Options,
                        PathRef FileName,
                        const tooling::CompileCommand &Command,
                        PrecompiledPreamble const *Preamble, StringRef Contents,
                        Position Pos, IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                        std::shared_ptr<PCHContainerOperations> PCHs,
                        clangd::Logger &Logger) {
  std::vector<const char *> ArgStrs;
  for (const auto &S : Command.CommandLine)
    ArgStrs.push_back(S.c_str());

  VFS->setCurrentWorkingDirectory(Command.Directory);

  std::unique_ptr<CompilerInvocation> CI;
  EmptyDiagsConsumer DummyDiagsConsumer;
  {
    IntrusiveRefCntPtr<DiagnosticsEngine> CommandLineDiagsEngine =
        CompilerInstance::createDiagnostics(new DiagnosticOptions,
                                            &DummyDiagsConsumer, false);
    CI = createCompilerInvocation(ArgStrs, CommandLineDiagsEngine, VFS);
  }
  assert(CI && "Couldn't create CompilerInvocation");

  std::unique_ptr<llvm::MemoryBuffer> ContentsBuffer =
      llvm::MemoryBuffer::getMemBufferCopy(Contents, FileName);

  // Attempt to reuse the PCH from precompiled preamble, if it was built.
  if (Preamble) {
    auto Bounds =
        ComputePreambleBounds(*CI->getLangOpts(), ContentsBuffer.get(), 0);
    if (!Preamble->CanReuse(*CI, ContentsBuffer.get(), Bounds, VFS.get()))
      Preamble = nullptr;
  }

  auto Clang = prepareCompilerInstance(
      std::move(CI), Preamble, std::move(ContentsBuffer), std::move(PCHs),
      std::move(VFS), DummyDiagsConsumer);
  auto &DiagOpts = Clang->getDiagnosticOpts();
  DiagOpts.IgnoreWarnings = true;

  auto &FrontendOpts = Clang->getFrontendOpts();
  FrontendOpts.SkipFunctionBodies = true;
  FrontendOpts.CodeCompleteOpts = Options;
  FrontendOpts.CodeCompletionAt.FileName = FileName;
  FrontendOpts.CodeCompletionAt.Line = Pos.line + 1;
  FrontendOpts.CodeCompletionAt.Column = Pos.character + 1;

  Clang->setCodeCompletionConsumer(Consumer.release());

  SyntaxOnlyAction Action;
  if (!Action.BeginSourceFile(*Clang, Clang->getFrontendOpts().Inputs[0])) {
    Logger.log("BeginSourceFile() failed when running codeComplete for " +
               FileName);
    return false;
  }
  if (!Action.Execute()) {
    Logger.log("Execute() failed when running codeComplete for " + FileName);
    return false;
  }

  Action.EndSourceFile();

  return true;
}

} // namespace

clangd::CodeCompleteOptions::CodeCompleteOptions(
    bool EnableSnippetsAndCodePatterns)
    : CodeCompleteOptions() {
  EnableSnippets = EnableSnippetsAndCodePatterns;
  IncludeCodePatterns = EnableSnippetsAndCodePatterns;
}

clangd::CodeCompleteOptions::CodeCompleteOptions(bool EnableSnippets,
                                                 bool IncludeCodePatterns,
                                                 bool IncludeMacros,
                                                 bool IncludeGlobals,
                                                 bool IncludeBriefComments)
    : EnableSnippets(EnableSnippets), IncludeCodePatterns(IncludeCodePatterns),
      IncludeMacros(IncludeMacros), IncludeGlobals(IncludeGlobals),
      IncludeBriefComments(IncludeBriefComments) {}

clang::CodeCompleteOptions clangd::CodeCompleteOptions::getClangCompleteOpts() {
  clang::CodeCompleteOptions Result;
  Result.IncludeCodePatterns = EnableSnippets && IncludeCodePatterns;
  Result.IncludeMacros = IncludeMacros;
  Result.IncludeGlobals = IncludeGlobals;
  Result.IncludeBriefComments = IncludeBriefComments;

  return Result;
}

std::vector<CompletionItem>
clangd::codeComplete(PathRef FileName, const tooling::CompileCommand &Command,
                     PrecompiledPreamble const *Preamble, StringRef Contents,
                     Position Pos, IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                     std::shared_ptr<PCHContainerOperations> PCHs,
                     clangd::CodeCompleteOptions Opts, clangd::Logger &Logger) {
  std::vector<CompletionItem> Results;
  std::unique_ptr<CodeCompleteConsumer> Consumer;
  clang::CodeCompleteOptions ClangCompleteOpts = Opts.getClangCompleteOpts();
  if (Opts.EnableSnippets) {
    Consumer = llvm::make_unique<SnippetCompletionItemsCollector>(
        ClangCompleteOpts, Results);
  } else {
    Consumer = llvm::make_unique<PlainTextCompletionItemsCollector>(
        ClangCompleteOpts, Results);
  }
  invokeCodeComplete(std::move(Consumer), ClangCompleteOpts, FileName, Command,
                     Preamble, Contents, Pos, std::move(VFS), std::move(PCHs),
                     Logger);
  return Results;
}

SignatureHelp
clangd::signatureHelp(PathRef FileName, const tooling::CompileCommand &Command,
                      PrecompiledPreamble const *Preamble, StringRef Contents,
                      Position Pos, IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                      std::shared_ptr<PCHContainerOperations> PCHs,
                      clangd::Logger &Logger) {
  SignatureHelp Result;
  clang::CodeCompleteOptions Options;
  Options.IncludeGlobals = false;
  Options.IncludeMacros = false;
  Options.IncludeCodePatterns = false;
  Options.IncludeBriefComments = true;
  invokeCodeComplete(llvm::make_unique<SignatureHelpCollector>(Options, Result),
                     Options, FileName, Command, Preamble, Contents, Pos,
                     std::move(VFS), std::move(PCHs), Logger);
  return Result;
}

void clangd::dumpAST(ParsedAST &AST, llvm::raw_ostream &OS) {
  AST.getASTContext().getTranslationUnitDecl()->dump(OS, true);
}

llvm::Optional<ParsedAST>
ParsedAST::Build(std::unique_ptr<clang::CompilerInvocation> CI,
                 const PrecompiledPreamble *Preamble,
                 ArrayRef<serialization::DeclID> PreambleDeclIDs,
                 std::unique_ptr<llvm::MemoryBuffer> Buffer,
                 std::shared_ptr<PCHContainerOperations> PCHs,
                 IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                 clangd::Logger &Logger) {

  std::vector<DiagWithFixIts> ASTDiags;
  StoreDiagsConsumer UnitDiagsConsumer(/*ref*/ ASTDiags);

  auto Clang = prepareCompilerInstance(
      std::move(CI), Preamble, std::move(Buffer), std::move(PCHs),
      std::move(VFS), /*ref*/ UnitDiagsConsumer);

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<CompilerInstance> CICleanup(
      Clang.get());

  auto Action = llvm::make_unique<ClangdFrontendAction>();
  const FrontendInputFile &MainInput = Clang->getFrontendOpts().Inputs[0];
  if (!Action->BeginSourceFile(*Clang, MainInput)) {
    Logger.log("BeginSourceFile() failed when building AST for " +
               MainInput.getFile());
    return llvm::None;
  }
  if (!Action->Execute())
    Logger.log("Execute() failed when building AST for " + MainInput.getFile());

  // UnitDiagsConsumer is local, we can not store it in CompilerInstance that
  // has a longer lifetime.
  Clang->getDiagnostics().setClient(new EmptyDiagsConsumer);

  std::vector<const Decl *> ParsedDecls = Action->takeTopLevelDecls();
  std::vector<serialization::DeclID> PendingDecls;
  if (Preamble) {
    PendingDecls.reserve(PreambleDeclIDs.size());
    PendingDecls.insert(PendingDecls.begin(), PreambleDeclIDs.begin(),
                        PreambleDeclIDs.end());
  }

  return ParsedAST(std::move(Clang), std::move(Action), std::move(ParsedDecls),
                   std::move(PendingDecls), std::move(ASTDiags));
}

namespace {

SourceLocation getMacroArgExpandedLocation(const SourceManager &Mgr,
                                           const FileEntry *FE,
                                           unsigned Offset) {
  SourceLocation FileLoc = Mgr.translateFileLineCol(FE, 1, 1);
  return Mgr.getMacroArgExpandedLocation(FileLoc.getLocWithOffset(Offset));
}

SourceLocation getMacroArgExpandedLocation(const SourceManager &Mgr,
                                           const FileEntry *FE, Position Pos) {
  SourceLocation InputLoc =
      Mgr.translateFileLineCol(FE, Pos.line + 1, Pos.character + 1);
  return Mgr.getMacroArgExpandedLocation(InputLoc);
}

/// Finds declarations locations that a given source location refers to.
class DeclarationLocationsFinder : public index::IndexDataConsumer {
  std::vector<Location> DeclarationLocations;
  const SourceLocation &SearchedLocation;
  const ASTContext &AST;
  Preprocessor &PP;

public:
  DeclarationLocationsFinder(raw_ostream &OS,
                             const SourceLocation &SearchedLocation,
                             ASTContext &AST, Preprocessor &PP)
      : SearchedLocation(SearchedLocation), AST(AST), PP(PP) {}

  std::vector<Location> takeLocations() {
    // Don't keep the same location multiple times.
    // This can happen when nodes in the AST are visited twice.
    std::sort(DeclarationLocations.begin(), DeclarationLocations.end());
    auto last =
        std::unique(DeclarationLocations.begin(), DeclarationLocations.end());
    DeclarationLocations.erase(last, DeclarationLocations.end());
    return std::move(DeclarationLocations);
  }

  bool
  handleDeclOccurence(const Decl *D, index::SymbolRoleSet Roles,
                      ArrayRef<index::SymbolRelation> Relations, FileID FID,
                      unsigned Offset,
                      index::IndexDataConsumer::ASTNodeInfo ASTNode) override {
    if (isSearchedLocation(FID, Offset)) {
      addDeclarationLocation(D->getSourceRange());
    }
    return true;
  }

private:
  bool isSearchedLocation(FileID FID, unsigned Offset) const {
    const SourceManager &SourceMgr = AST.getSourceManager();
    return SourceMgr.getFileOffset(SearchedLocation) == Offset &&
           SourceMgr.getFileID(SearchedLocation) == FID;
  }

  void addDeclarationLocation(const SourceRange &ValSourceRange) {
    const SourceManager &SourceMgr = AST.getSourceManager();
    const LangOptions &LangOpts = AST.getLangOpts();
    SourceLocation LocStart = ValSourceRange.getBegin();
    SourceLocation LocEnd = Lexer::getLocForEndOfToken(ValSourceRange.getEnd(),
                                                       0, SourceMgr, LangOpts);
    Position Begin;
    Begin.line = SourceMgr.getSpellingLineNumber(LocStart) - 1;
    Begin.character = SourceMgr.getSpellingColumnNumber(LocStart) - 1;
    Position End;
    End.line = SourceMgr.getSpellingLineNumber(LocEnd) - 1;
    End.character = SourceMgr.getSpellingColumnNumber(LocEnd) - 1;
    Range R = {Begin, End};
    Location L;
    if (const FileEntry *F =
            SourceMgr.getFileEntryForID(SourceMgr.getFileID(LocStart))) {
      StringRef FilePath = F->tryGetRealPathName();
      if (FilePath.empty())
        FilePath = F->getName();
      L.uri = URI::fromFile(FilePath);
      L.range = R;
      DeclarationLocations.push_back(L);
    }
  }

  void finish() override {
    // Also handle possible macro at the searched location.
    Token Result;
    if (!Lexer::getRawToken(SearchedLocation, Result, AST.getSourceManager(),
                            AST.getLangOpts(), false)) {
      if (Result.is(tok::raw_identifier)) {
        PP.LookUpIdentifierInfo(Result);
      }
      IdentifierInfo *IdentifierInfo = Result.getIdentifierInfo();
      if (IdentifierInfo && IdentifierInfo->hadMacroDefinition()) {
        std::pair<FileID, unsigned int> DecLoc =
            AST.getSourceManager().getDecomposedExpansionLoc(SearchedLocation);
        // Get the definition just before the searched location so that a macro
        // referenced in a '#undef MACRO' can still be found.
        SourceLocation BeforeSearchedLocation = getMacroArgExpandedLocation(
            AST.getSourceManager(),
            AST.getSourceManager().getFileEntryForID(DecLoc.first),
            DecLoc.second - 1);
        MacroDefinition MacroDef =
            PP.getMacroDefinitionAtLoc(IdentifierInfo, BeforeSearchedLocation);
        MacroInfo *MacroInf = MacroDef.getMacroInfo();
        if (MacroInf) {
          addDeclarationLocation(SourceRange(MacroInf->getDefinitionLoc(),
                                             MacroInf->getDefinitionEndLoc()));
        }
      }
    }
  }
};

} // namespace

std::vector<Location> clangd::findDefinitions(ParsedAST &AST, Position Pos,
                                              clangd::Logger &Logger) {
  const SourceManager &SourceMgr = AST.getASTContext().getSourceManager();
  const FileEntry *FE = SourceMgr.getFileEntryForID(SourceMgr.getMainFileID());
  if (!FE)
    return {};

  SourceLocation SourceLocationBeg = getBeginningOfIdentifier(AST, Pos, FE);

  auto DeclLocationsFinder = std::make_shared<DeclarationLocationsFinder>(
      llvm::errs(), SourceLocationBeg, AST.getASTContext(),
      AST.getPreprocessor());
  index::IndexingOptions IndexOpts;
  IndexOpts.SystemSymbolFilter =
      index::IndexingOptions::SystemSymbolFilterKind::All;
  IndexOpts.IndexFunctionLocals = true;

  indexTopLevelDecls(AST.getASTContext(), AST.getTopLevelDecls(),
                     DeclLocationsFinder, IndexOpts);

  return DeclLocationsFinder->takeLocations();
}

void ParsedAST::ensurePreambleDeclsDeserialized() {
  if (PendingTopLevelDecls.empty())
    return;

  std::vector<const Decl *> Resolved;
  Resolved.reserve(PendingTopLevelDecls.size());

  ExternalASTSource &Source = *getASTContext().getExternalSource();
  for (serialization::DeclID TopLevelDecl : PendingTopLevelDecls) {
    // Resolve the declaration ID to an actual declaration, possibly
    // deserializing the declaration in the process.
    if (Decl *D = Source.GetExternalDecl(TopLevelDecl))
      Resolved.push_back(D);
  }

  TopLevelDecls.reserve(TopLevelDecls.size() + PendingTopLevelDecls.size());
  TopLevelDecls.insert(TopLevelDecls.begin(), Resolved.begin(), Resolved.end());

  PendingTopLevelDecls.clear();
}

ParsedAST::ParsedAST(ParsedAST &&Other) = default;

ParsedAST &ParsedAST::operator=(ParsedAST &&Other) = default;

ParsedAST::~ParsedAST() {
  if (Action) {
    Action->EndSourceFile();
  }
}

ASTContext &ParsedAST::getASTContext() { return Clang->getASTContext(); }

const ASTContext &ParsedAST::getASTContext() const {
  return Clang->getASTContext();
}

Preprocessor &ParsedAST::getPreprocessor() { return Clang->getPreprocessor(); }

const Preprocessor &ParsedAST::getPreprocessor() const {
  return Clang->getPreprocessor();
}

ArrayRef<const Decl *> ParsedAST::getTopLevelDecls() {
  ensurePreambleDeclsDeserialized();
  return TopLevelDecls;
}

const std::vector<DiagWithFixIts> &ParsedAST::getDiagnostics() const {
  return Diags;
}

ParsedAST::ParsedAST(std::unique_ptr<CompilerInstance> Clang,
                     std::unique_ptr<FrontendAction> Action,
                     std::vector<const Decl *> TopLevelDecls,
                     std::vector<serialization::DeclID> PendingTopLevelDecls,
                     std::vector<DiagWithFixIts> Diags)
    : Clang(std::move(Clang)), Action(std::move(Action)),
      Diags(std::move(Diags)), TopLevelDecls(std::move(TopLevelDecls)),
      PendingTopLevelDecls(std::move(PendingTopLevelDecls)) {
  assert(this->Clang);
  assert(this->Action);
}

ParsedASTWrapper::ParsedASTWrapper(ParsedASTWrapper &&Wrapper)
    : AST(std::move(Wrapper.AST)) {}

ParsedASTWrapper::ParsedASTWrapper(llvm::Optional<ParsedAST> AST)
    : AST(std::move(AST)) {}

PreambleData::PreambleData(PrecompiledPreamble Preamble,
                           std::vector<serialization::DeclID> TopLevelDeclIDs,
                           std::vector<DiagWithFixIts> Diags)
    : Preamble(std::move(Preamble)),
      TopLevelDeclIDs(std::move(TopLevelDeclIDs)), Diags(std::move(Diags)) {}

std::shared_ptr<CppFile>
CppFile::Create(PathRef FileName, tooling::CompileCommand Command,
                std::shared_ptr<PCHContainerOperations> PCHs,
                clangd::Logger &Logger) {
  return std::shared_ptr<CppFile>(
      new CppFile(FileName, std::move(Command), std::move(PCHs), Logger));
}

CppFile::CppFile(PathRef FileName, tooling::CompileCommand Command,
                 std::shared_ptr<PCHContainerOperations> PCHs,
                 clangd::Logger &Logger)
    : FileName(FileName), Command(std::move(Command)), RebuildCounter(0),
      RebuildInProgress(false), PCHs(std::move(PCHs)), Logger(Logger) {

  std::lock_guard<std::mutex> Lock(Mutex);
  LatestAvailablePreamble = nullptr;
  PreamblePromise.set_value(nullptr);
  PreambleFuture = PreamblePromise.get_future();

  ASTPromise.set_value(std::make_shared<ParsedASTWrapper>(llvm::None));
  ASTFuture = ASTPromise.get_future();
}

void CppFile::cancelRebuild() { deferCancelRebuild()(); }

UniqueFunction<void()> CppFile::deferCancelRebuild() {
  std::unique_lock<std::mutex> Lock(Mutex);
  // Cancel an ongoing rebuild, if any, and wait for it to finish.
  unsigned RequestRebuildCounter = ++this->RebuildCounter;
  // Rebuild asserts that futures aren't ready if rebuild is cancelled.
  // We want to keep this invariant.
  if (futureIsReady(PreambleFuture)) {
    PreamblePromise = std::promise<std::shared_ptr<const PreambleData>>();
    PreambleFuture = PreamblePromise.get_future();
  }
  if (futureIsReady(ASTFuture)) {
    ASTPromise = std::promise<std::shared_ptr<ParsedASTWrapper>>();
    ASTFuture = ASTPromise.get_future();
  }

  Lock.unlock();
  // Notify about changes to RebuildCounter.
  RebuildCond.notify_all();

  std::shared_ptr<CppFile> That = shared_from_this();
  return [That, RequestRebuildCounter]() {
    std::unique_lock<std::mutex> Lock(That->Mutex);
    CppFile *This = &*That;
    This->RebuildCond.wait(Lock, [This, RequestRebuildCounter]() {
      return !This->RebuildInProgress ||
             This->RebuildCounter != RequestRebuildCounter;
    });

    // This computation got cancelled itself, do nothing.
    if (This->RebuildCounter != RequestRebuildCounter)
      return;

    // Set empty results for Promises.
    That->PreamblePromise.set_value(nullptr);
    That->ASTPromise.set_value(std::make_shared<ParsedASTWrapper>(llvm::None));
  };
}

llvm::Optional<std::vector<DiagWithFixIts>>
CppFile::rebuild(StringRef NewContents,
                 IntrusiveRefCntPtr<vfs::FileSystem> VFS) {
  return deferRebuild(NewContents, std::move(VFS))();
}

UniqueFunction<llvm::Optional<std::vector<DiagWithFixIts>>()>
CppFile::deferRebuild(StringRef NewContents,
                      IntrusiveRefCntPtr<vfs::FileSystem> VFS) {
  std::shared_ptr<const PreambleData> OldPreamble;
  std::shared_ptr<PCHContainerOperations> PCHs;
  unsigned RequestRebuildCounter;
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    // Increase RebuildCounter to cancel all ongoing FinishRebuild operations.
    // They will try to exit as early as possible and won't call set_value on
    // our promises.
    RequestRebuildCounter = ++this->RebuildCounter;
    PCHs = this->PCHs;

    // Remember the preamble to be used during rebuild.
    OldPreamble = this->LatestAvailablePreamble;
    // Setup std::promises and std::futures for Preamble and AST. Corresponding
    // futures will wait until the rebuild process is finished.
    if (futureIsReady(this->PreambleFuture)) {
      this->PreamblePromise =
          std::promise<std::shared_ptr<const PreambleData>>();
      this->PreambleFuture = this->PreamblePromise.get_future();
    }
    if (futureIsReady(this->ASTFuture)) {
      this->ASTPromise = std::promise<std::shared_ptr<ParsedASTWrapper>>();
      this->ASTFuture = this->ASTPromise.get_future();
    }
  } // unlock Mutex.
  // Notify about changes to RebuildCounter.
  RebuildCond.notify_all();

  // A helper to function to finish the rebuild. May be run on a different
  // thread.

  // Don't let this CppFile die before rebuild is finished.
  std::shared_ptr<CppFile> That = shared_from_this();
  auto FinishRebuild = [OldPreamble, VFS, RequestRebuildCounter, PCHs,
                        That](std::string NewContents)
      -> llvm::Optional<std::vector<DiagWithFixIts>> {
    // Only one execution of this method is possible at a time.
    // RebuildGuard will wait for any ongoing rebuilds to finish and will put us
    // into a state for doing a rebuild.
    RebuildGuard Rebuild(*That, RequestRebuildCounter);
    if (Rebuild.wasCancelledBeforeConstruction())
      return llvm::None;

    std::vector<const char *> ArgStrs;
    for (const auto &S : That->Command.CommandLine)
      ArgStrs.push_back(S.c_str());

    VFS->setCurrentWorkingDirectory(That->Command.Directory);

    std::unique_ptr<CompilerInvocation> CI;
    {
      // FIXME(ibiryukov): store diagnostics from CommandLine when we start
      // reporting them.
      EmptyDiagsConsumer CommandLineDiagsConsumer;
      IntrusiveRefCntPtr<DiagnosticsEngine> CommandLineDiagsEngine =
          CompilerInstance::createDiagnostics(new DiagnosticOptions,
                                              &CommandLineDiagsConsumer, false);
      CI = createCompilerInvocation(ArgStrs, CommandLineDiagsEngine, VFS);
    }
    assert(CI && "Couldn't create CompilerInvocation");

    std::unique_ptr<llvm::MemoryBuffer> ContentsBuffer =
        llvm::MemoryBuffer::getMemBufferCopy(NewContents, That->FileName);

    // A helper function to rebuild the preamble or reuse the existing one. Does
    // not mutate any fields, only does the actual computation.
    auto DoRebuildPreamble = [&]() -> std::shared_ptr<const PreambleData> {
      auto Bounds =
          ComputePreambleBounds(*CI->getLangOpts(), ContentsBuffer.get(), 0);
      if (OldPreamble && OldPreamble->Preamble.CanReuse(
                             *CI, ContentsBuffer.get(), Bounds, VFS.get())) {
        return OldPreamble;
      }

      trace::Span Tracer(llvm::Twine("Preamble: ") + That->FileName);
      std::vector<DiagWithFixIts> PreambleDiags;
      StoreDiagsConsumer PreambleDiagnosticsConsumer(/*ref*/ PreambleDiags);
      IntrusiveRefCntPtr<DiagnosticsEngine> PreambleDiagsEngine =
          CompilerInstance::createDiagnostics(
              &CI->getDiagnosticOpts(), &PreambleDiagnosticsConsumer, false);
      CppFilePreambleCallbacks SerializedDeclsCollector;
      auto BuiltPreamble = PrecompiledPreamble::Build(
          *CI, ContentsBuffer.get(), Bounds, *PreambleDiagsEngine, VFS, PCHs,
          SerializedDeclsCollector);

      if (BuiltPreamble) {
        return std::make_shared<PreambleData>(
            std::move(*BuiltPreamble),
            SerializedDeclsCollector.takeTopLevelDeclIDs(),
            std::move(PreambleDiags));
      } else {
        return nullptr;
      }
    };

    // Compute updated Preamble.
    std::shared_ptr<const PreambleData> NewPreamble = DoRebuildPreamble();
    // Publish the new Preamble.
    {
      std::lock_guard<std::mutex> Lock(That->Mutex);
      // We always set LatestAvailablePreamble to the new value, hoping that it
      // will still be usable in the further requests.
      That->LatestAvailablePreamble = NewPreamble;
      if (RequestRebuildCounter != That->RebuildCounter)
        return llvm::None; // Our rebuild request was cancelled, do nothing.
      That->PreamblePromise.set_value(NewPreamble);
    } // unlock Mutex

    // Prepare the Preamble and supplementary data for rebuilding AST.
    const PrecompiledPreamble *PreambleForAST = nullptr;
    ArrayRef<serialization::DeclID> SerializedPreambleDecls = llvm::None;
    std::vector<DiagWithFixIts> Diagnostics;
    if (NewPreamble) {
      PreambleForAST = &NewPreamble->Preamble;
      SerializedPreambleDecls = NewPreamble->TopLevelDeclIDs;
      Diagnostics.insert(Diagnostics.begin(), NewPreamble->Diags.begin(),
                         NewPreamble->Diags.end());
    }

    // Compute updated AST.
    llvm::Optional<ParsedAST> NewAST;
    {
      trace::Span Tracer(llvm::Twine("Build: ") + That->FileName);
      NewAST = ParsedAST::Build(
          std::move(CI), PreambleForAST, SerializedPreambleDecls,
          std::move(ContentsBuffer), PCHs, VFS, That->Logger);
    }

    if (NewAST) {
      Diagnostics.insert(Diagnostics.end(), NewAST->getDiagnostics().begin(),
                         NewAST->getDiagnostics().end());
    } else {
      // Don't report even Preamble diagnostics if we coulnd't build AST.
      Diagnostics.clear();
    }

    // Publish the new AST.
    {
      std::lock_guard<std::mutex> Lock(That->Mutex);
      if (RequestRebuildCounter != That->RebuildCounter)
        return Diagnostics; // Our rebuild request was cancelled, don't set
                            // ASTPromise.

      That->ASTPromise.set_value(
          std::make_shared<ParsedASTWrapper>(std::move(NewAST)));
    } // unlock Mutex

    return Diagnostics;
  };

  return BindWithForward(FinishRebuild, NewContents.str());
}

std::shared_future<std::shared_ptr<const PreambleData>>
CppFile::getPreamble() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return PreambleFuture;
}

std::shared_ptr<const PreambleData> CppFile::getPossiblyStalePreamble() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return LatestAvailablePreamble;
}

std::shared_future<std::shared_ptr<ParsedASTWrapper>> CppFile::getAST() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return ASTFuture;
}

tooling::CompileCommand const &CppFile::getCompileCommand() const {
  return Command;
}

CppFile::RebuildGuard::RebuildGuard(CppFile &File,
                                    unsigned RequestRebuildCounter)
    : File(File), RequestRebuildCounter(RequestRebuildCounter) {
  std::unique_lock<std::mutex> Lock(File.Mutex);
  WasCancelledBeforeConstruction = File.RebuildCounter != RequestRebuildCounter;
  if (WasCancelledBeforeConstruction)
    return;

  File.RebuildCond.wait(Lock, [&File, RequestRebuildCounter]() {
    return !File.RebuildInProgress ||
           File.RebuildCounter != RequestRebuildCounter;
  });

  WasCancelledBeforeConstruction = File.RebuildCounter != RequestRebuildCounter;
  if (WasCancelledBeforeConstruction)
    return;

  File.RebuildInProgress = true;
}

bool CppFile::RebuildGuard::wasCancelledBeforeConstruction() const {
  return WasCancelledBeforeConstruction;
}

CppFile::RebuildGuard::~RebuildGuard() {
  if (WasCancelledBeforeConstruction)
    return;

  std::unique_lock<std::mutex> Lock(File.Mutex);
  assert(File.RebuildInProgress);
  File.RebuildInProgress = false;

  if (File.RebuildCounter == RequestRebuildCounter) {
    // Our rebuild request was successful.
    assert(futureIsReady(File.ASTFuture));
    assert(futureIsReady(File.PreambleFuture));
  } else {
    // Our rebuild request was cancelled, because further reparse was requested.
    assert(!futureIsReady(File.ASTFuture));
    assert(!futureIsReady(File.PreambleFuture));
  }

  Lock.unlock();
  File.RebuildCond.notify_all();
}

SourceLocation clangd::getBeginningOfIdentifier(ParsedAST &Unit,
                                                const Position &Pos,
                                                const FileEntry *FE) {
  // The language server protocol uses zero-based line and column numbers.
  // Clang uses one-based numbers.

  const ASTContext &AST = Unit.getASTContext();
  const SourceManager &SourceMgr = AST.getSourceManager();

  SourceLocation InputLocation =
      getMacroArgExpandedLocation(SourceMgr, FE, Pos);
  if (Pos.character == 0) {
    return InputLocation;
  }

  // This handle cases where the position is in the middle of a token or right
  // after the end of a token. In theory we could just use GetBeginningOfToken
  // to find the start of the token at the input position, but this doesn't
  // work when right after the end, i.e. foo|.
  // So try to go back by one and see if we're still inside the an identifier
  // token. If so, Take the beginning of this token.
  // (It should be the same identifier because you can't have two adjacent
  // identifiers without another token in between.)
  SourceLocation PeekBeforeLocation = getMacroArgExpandedLocation(
      SourceMgr, FE, Position{Pos.line, Pos.character - 1});
  Token Result;
  if (Lexer::getRawToken(PeekBeforeLocation, Result, SourceMgr,
                         AST.getLangOpts(), false)) {
    // getRawToken failed, just use InputLocation.
    return InputLocation;
  }

  if (Result.is(tok::raw_identifier)) {
    return Lexer::GetBeginningOfToken(PeekBeforeLocation, SourceMgr,
                                      AST.getLangOpts());
  }

  return InputLocation;
}
