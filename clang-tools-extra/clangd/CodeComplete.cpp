//===--- CodeComplete.cpp ---------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// AST-based completions are provided using the completion hooks in Sema.
//
// Signature help works in a similar way as code completion, but it is simpler
// as there are typically fewer candidates.
//
//===---------------------------------------------------------------------===//

#include "CodeComplete.h"
#include "Compiler.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/Sema.h"
#include <queue>

namespace clang {
namespace clangd {
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

/// A scored code completion result.
/// It may be promoted to a CompletionItem if it's among the top-ranked results.
struct CompletionCandidate {
  CompletionCandidate(CodeCompletionResult &Result)
      : Result(&Result), Score(score(Result)) {}

  CodeCompletionResult *Result;
  float Score; // 0 to 1, higher is better.

  // Comparison reflects rank: better candidates are smaller.
  bool operator<(const CompletionCandidate &C) const {
    if (Score != C.Score)
      return Score > C.Score;
    return *Result < *C.Result;
  }

  // Returns a string that sorts in the same order as operator<, for LSP.
  // Conceptually, this is [-Score, Name]. We convert -Score to an integer, and
  // hex-encode it for readability. Example: [0.5, "foo"] -> "41000000foo"
  std::string sortText() const {
    std::string S, NameStorage;
    llvm::raw_string_ostream OS(S);
    write_hex(OS, encodeFloat(-Score), llvm::HexPrintStyle::Lower,
              /*Width=*/2 * sizeof(Score));
    OS << Result->getOrderedName(NameStorage);
    return OS.str();
  }

private:
  static float score(const CodeCompletionResult &Result) {
    // Priority 80 is a really bad score.
    float Score = 1 - std::min<float>(80, Result.Priority) / 80;

    switch (static_cast<CXAvailabilityKind>(Result.Availability)) {
    case CXAvailability_Available:
      // No penalty.
      break;
    case CXAvailability_Deprecated:
      Score *= 0.1f;
      break;
    case CXAvailability_NotAccessible:
    case CXAvailability_NotAvailable:
      Score = 0;
      break;
    }
    return Score;
  }

  // Produces an integer that sorts in the same order as F.
  // That is: a < b <==> encodeFloat(a) < encodeFloat(b).
  static uint32_t encodeFloat(float F) {
    static_assert(std::numeric_limits<float>::is_iec559, "");
    static_assert(sizeof(float) == sizeof(uint32_t), "");
    constexpr uint32_t TopBit = ~(~uint32_t{0} >> 1);

    // Get the bits of the float. Endianness is the same as for integers.
    uint32_t U;
    memcpy(&U, &F, sizeof(float));
    // IEEE 754 floats compare like sign-magnitude integers.
    if (U & TopBit)    // Negative float.
      return 0 - U;    // Map onto the low half of integers, order reversed.
    return U + TopBit; // Positive floats map onto the high half of integers.
  }
};

class CompletionItemsCollector : public CodeCompleteConsumer {
public:
  CompletionItemsCollector(const CodeCompleteOptions &CodeCompleteOpts,
                           CompletionList &Items)
      : CodeCompleteConsumer(CodeCompleteOpts.getClangCompleteOpts(),
                             /*OutputIsBinary=*/false),
        ClangdOpts(CodeCompleteOpts), Items(Items),
        Allocator(std::make_shared<clang::GlobalCodeCompletionAllocator>()),
        CCTUInfo(Allocator) {}

  void ProcessCodeCompleteResults(Sema &S, CodeCompletionContext Context,
                                  CodeCompletionResult *Results,
                                  unsigned NumResults) override final {
    StringRef Filter = S.getPreprocessor().getCodeCompletionFilter();
    std::priority_queue<CompletionCandidate> Candidates;
    for (unsigned I = 0; I < NumResults; ++I) {
      auto &Result = Results[I];
      if (!ClangdOpts.IncludeIneligibleResults &&
          (Result.Availability == CXAvailability_NotAvailable ||
           Result.Availability == CXAvailability_NotAccessible))
        continue;
      if (!Filter.empty() && !fuzzyMatch(S, Context, Filter, Result))
        continue;
      Candidates.emplace(Result);
      if (ClangdOpts.Limit && Candidates.size() > ClangdOpts.Limit) {
        Candidates.pop();
        Items.isIncomplete = true;
      }
    }
    while (!Candidates.empty()) {
      auto &Candidate = Candidates.top();
      const auto *CCS = Candidate.Result->CreateCodeCompletionString(
          S, Context, *Allocator, CCTUInfo,
          CodeCompleteOpts.IncludeBriefComments);
      assert(CCS && "Expected the CodeCompletionString to be non-null");
      Items.items.push_back(ProcessCodeCompleteResult(Candidate, *CCS));
      Candidates.pop();
    }
    std::reverse(Items.items.begin(), Items.items.end());
  }

  GlobalCodeCompletionAllocator &getAllocator() override { return *Allocator; }

  CodeCompletionTUInfo &getCodeCompletionTUInfo() override { return CCTUInfo; }

private:
  bool fuzzyMatch(Sema &S, const CodeCompletionContext &CCCtx, StringRef Filter,
                  CodeCompletionResult Result) {
    switch (Result.Kind) {
    case CodeCompletionResult::RK_Declaration:
      if (auto *ID = Result.Declaration->getIdentifier())
        return fuzzyMatch(Filter, ID->getName());
      break;
    case CodeCompletionResult::RK_Keyword:
      return fuzzyMatch(Filter, Result.Keyword);
    case CodeCompletionResult::RK_Macro:
      return fuzzyMatch(Filter, Result.Macro->getName());
    case CodeCompletionResult::RK_Pattern:
      return fuzzyMatch(Filter, Result.Pattern->getTypedText());
    }
    auto *CCS = Result.CreateCodeCompletionString(
        S, CCCtx, *Allocator, CCTUInfo, /*IncludeBriefComments=*/false);
    return fuzzyMatch(Filter, CCS->getTypedText());
  }

  // Checks whether Target matches the Filter.
  // Currently just requires a case-insensitive subsequence match.
  // FIXME: make stricter and word-based: 'unique_ptr' should not match 'que'.
  // FIXME: return a score to be incorporated into ranking.
  static bool fuzzyMatch(StringRef Filter, StringRef Target) {
    size_t TPos = 0;
    for (char C : Filter) {
      TPos = Target.find_lower(C, TPos);
      if (TPos == StringRef::npos)
        return false;
    }
    return true;
  }

  CompletionItem
  ProcessCodeCompleteResult(const CompletionCandidate &Candidate,
                            const CodeCompletionString &CCS) const {

    // Adjust this to InsertTextFormat::Snippet iff we encounter a
    // CK_Placeholder chunk in SnippetCompletionItemsCollector.
    CompletionItem Item;
    Item.insertTextFormat = InsertTextFormat::PlainText;

    Item.documentation = getDocumentation(CCS);
    Item.sortText = Candidate.sortText();

    // Fill in the label, detail, insertText and filterText fields of the
    // CompletionItem.
    ProcessChunks(CCS, Item);

    // Fill in the kind field of the CompletionItem.
    Item.kind = getKind(Candidate.Result->Kind, Candidate.Result->CursorKind);

    return Item;
  }

  virtual void ProcessChunks(const CodeCompletionString &CCS,
                             CompletionItem &Item) const = 0;

  CodeCompleteOptions ClangdOpts;
  CompletionList &Items;
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
  PlainTextCompletionItemsCollector(const CodeCompleteOptions &CodeCompleteOpts,
                                    CompletionList &Items)
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
  SnippetCompletionItemsCollector(const CodeCompleteOptions &CodeCompleteOpts,
                                  CompletionList &Items)
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

bool invokeCodeComplete(const Context &Ctx,
                        std::unique_ptr<CodeCompleteConsumer> Consumer,
                        const clang::CodeCompleteOptions &Options,
                        PathRef FileName,
                        const tooling::CompileCommand &Command,
                        PrecompiledPreamble const *Preamble, StringRef Contents,
                        Position Pos, IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                        std::shared_ptr<PCHContainerOperations> PCHs) {
  std::vector<const char *> ArgStrs;
  for (const auto &S : Command.CommandLine)
    ArgStrs.push_back(S.c_str());

  VFS->setCurrentWorkingDirectory(Command.Directory);

  IgnoreDiagnostics DummyDiagsConsumer;
  auto CI = createInvocationFromCommandLine(
      ArgStrs,
      CompilerInstance::createDiagnostics(new DiagnosticOptions,
                                          &DummyDiagsConsumer, false),
      VFS);
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
    log(Ctx,
        "BeginSourceFile() failed when running codeComplete for " + FileName);
    return false;
  }
  if (!Action.Execute()) {
    log(Ctx, "Execute() failed when running codeComplete for " + FileName);
    return false;
  }

  Action.EndSourceFile();

  return true;
}

} // namespace

clang::CodeCompleteOptions CodeCompleteOptions::getClangCompleteOpts() const {
  clang::CodeCompleteOptions Result;
  Result.IncludeCodePatterns = EnableSnippets && IncludeCodePatterns;
  Result.IncludeMacros = IncludeMacros;
  Result.IncludeGlobals = IncludeGlobals;
  Result.IncludeBriefComments = IncludeBriefComments;

  return Result;
}

CompletionList codeComplete(const Context &Ctx, PathRef FileName,
                            const tooling::CompileCommand &Command,
                            PrecompiledPreamble const *Preamble,
                            StringRef Contents, Position Pos,
                            IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                            std::shared_ptr<PCHContainerOperations> PCHs,
                            CodeCompleteOptions Opts) {
  CompletionList Results;
  std::unique_ptr<CodeCompleteConsumer> Consumer;
  if (Opts.EnableSnippets) {
    Consumer =
        llvm::make_unique<SnippetCompletionItemsCollector>(Opts, Results);
  } else {
    Consumer =
        llvm::make_unique<PlainTextCompletionItemsCollector>(Opts, Results);
  }
  invokeCodeComplete(Ctx, std::move(Consumer), Opts.getClangCompleteOpts(),
                     FileName, Command, Preamble, Contents, Pos, std::move(VFS),
                     std::move(PCHs));
  return Results;
}

SignatureHelp signatureHelp(const Context &Ctx, PathRef FileName,
                            const tooling::CompileCommand &Command,
                            PrecompiledPreamble const *Preamble,
                            StringRef Contents, Position Pos,
                            IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                            std::shared_ptr<PCHContainerOperations> PCHs) {
  SignatureHelp Result;
  clang::CodeCompleteOptions Options;
  Options.IncludeGlobals = false;
  Options.IncludeMacros = false;
  Options.IncludeCodePatterns = false;
  Options.IncludeBriefComments = true;
  invokeCodeComplete(Ctx,
                     llvm::make_unique<SignatureHelpCollector>(Options, Result),
                     Options, FileName, Command, Preamble, Contents, Pos,
                     std::move(VFS), std::move(PCHs));
  return Result;
}

} // namespace clangd
} // namespace clang
