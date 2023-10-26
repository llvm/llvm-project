//===--- CodeCompletionStrings.cpp -------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodeCompletionStrings.h"
#include "clang-c/Index.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RawCommentList.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/JSON.h"
#include <limits>
#include <utility>

namespace clang {
namespace clangd {
namespace {

bool isInformativeQualifierChunk(CodeCompletionString::Chunk const &Chunk) {
  return Chunk.Kind == CodeCompletionString::CK_Informative &&
         llvm::StringRef(Chunk.Text).endswith("::");
}

void appendEscapeSnippet(const llvm::StringRef Text, std::string *Out) {
  for (const auto Character : Text) {
    if (Character == '$' || Character == '}' || Character == '\\')
      Out->push_back('\\');
    Out->push_back(Character);
  }
}

void appendOptionalChunk(const CodeCompletionString &CCS, std::string *Out) {
  for (const CodeCompletionString::Chunk &C : CCS) {
    switch (C.Kind) {
    case CodeCompletionString::CK_Optional:
      assert(C.Optional &&
             "Expected the optional code completion string to be non-null.");
      appendOptionalChunk(*C.Optional, Out);
      break;
    default:
      *Out += C.Text;
      break;
    }
  }
}

bool looksLikeDocComment(llvm::StringRef CommentText) {
  // We don't report comments that only contain "special" chars.
  // This avoids reporting various delimiters, like:
  //   =================
  //   -----------------
  //   *****************
  return CommentText.find_first_not_of("/*-= \t\r\n") != llvm::StringRef::npos;
}

// Determine whether the completion string should be patched
// to replace the last placeholder with $0.
bool shouldPatchPlaceholder0(CodeCompletionResult::ResultKind ResultKind,
                             CXCursorKind CursorKind) {
  bool CompletingPattern = ResultKind == CodeCompletionResult::RK_Pattern;

  if (!CompletingPattern)
    return false;

  // If the result kind of CodeCompletionResult(CCR) is `RK_Pattern`, it doesn't
  // always mean we're completing a chunk of statements.  Constructors defined
  // in base class, for example, are considered as a type of pattern, with the
  // cursor type set to CXCursor_Constructor.
  if (CursorKind == CXCursorKind::CXCursor_Constructor ||
      CursorKind == CXCursorKind::CXCursor_Destructor)
    return false;

  return true;
}

} // namespace

std::string getDocComment(const ASTContext &Ctx,
                          const CodeCompletionResult &Result,
                          bool CommentsFromHeaders) {
  // FIXME: clang's completion also returns documentation for RK_Pattern if they
  // contain a pattern for ObjC properties. Unfortunately, there is no API to
  // get this declaration, so we don't show documentation in that case.
  if (Result.Kind != CodeCompletionResult::RK_Declaration)
    return "";
  return Result.getDeclaration() ? getDeclComment(Ctx, *Result.getDeclaration())
                                 : "";
}

std::string getDeclComment(const ASTContext &Ctx, const NamedDecl &Decl) {
  if (isa<NamespaceDecl>(Decl)) {
    // Namespaces often have too many redecls for any particular redecl comment
    // to be useful. Moreover, we often confuse file headers or generated
    // comments with namespace comments. Therefore we choose to just ignore
    // the comments for namespaces.
    return "";
  }
  const RawComment *RC = getCompletionComment(Ctx, &Decl);
  if (!RC)
    return "";
  // Sanity check that the comment does not come from the PCH. We choose to not
  // write them into PCH, because they are racy and slow to load.
  assert(!Ctx.getSourceManager().isLoadedSourceLocation(RC->getBeginLoc()));
  std::string Doc =
      RC->getFormattedText(Ctx.getSourceManager(), Ctx.getDiagnostics());
  if (!looksLikeDocComment(Doc))
    return "";
  // Clang requires source to be UTF-8, but doesn't enforce this in comments.
  if (!llvm::json::isUTF8(Doc))
    Doc = llvm::json::fixUTF8(Doc);
  return Doc;
}

void getSignature(const CodeCompletionString &CCS, std::string *Signature,
                  std::string *Snippet,
                  CodeCompletionResult::ResultKind ResultKind,
                  CXCursorKind CursorKind, bool IncludeFunctionArguments,
                  std::string *RequiredQualifiers) {
  // Placeholder with this index will be $0 to mark final cursor position.
  // Usually we do not add $0, so the cursor is placed at end of completed text.
  unsigned CursorSnippetArg = std::numeric_limits<unsigned>::max();

  // If the snippet contains a group of statements, we replace the
  // last placeholder with $0 to leave the cursor there, e.g.
  //    namespace ${1:name} {
  //      ${0:decls}
  //    }
  // We try to identify such cases using the ResultKind and CursorKind.
  if (shouldPatchPlaceholder0(ResultKind, CursorKind)) {
    CursorSnippetArg =
        llvm::count_if(CCS, [](const CodeCompletionString::Chunk &C) {
          return C.Kind == CodeCompletionString::CK_Placeholder;
        });
  }
  unsigned SnippetArg = 0;
  bool HadObjCArguments = false;
  bool HadInformativeChunks = false;

  std::optional<unsigned> TruncateSnippetAt;
  for (const auto &Chunk : CCS) {
    // Informative qualifier chunks only clutter completion results, skip
    // them.
    if (isInformativeQualifierChunk(Chunk))
      continue;

    switch (Chunk.Kind) {
    case CodeCompletionString::CK_TypedText:
      // The typed-text chunk is the actual name. We don't record this chunk.
      // C++:
      //   In general our string looks like <qualifiers><name><signature>.
      //   So once we see the name, any text we recorded so far should be
      //   reclassified as qualifiers.
      //
      // Objective-C:
      //   Objective-C methods expressions may have multiple typed-text chunks,
      //   so we must treat them carefully. For Objective-C methods, all
      //   typed-text and informative chunks will end in ':' (unless there are
      //   no arguments, in which case we can safely treat them as C++).
      //
      //   Completing a method declaration itself (not a method expression) is
      //   similar except that we use the `RequiredQualifiers` to store the
      //   text before the selector, e.g. `- (void)`.
      if (!llvm::StringRef(Chunk.Text).endswith(":")) { // Treat as C++.
        if (RequiredQualifiers)
          *RequiredQualifiers = std::move(*Signature);
        Signature->clear();
        Snippet->clear();
      } else { // Objective-C method with args.
        // If this is the first TypedText to the Objective-C method, discard any
        // text that we've previously seen (such as previous parameter selector,
        // which will be marked as Informative text).
        //
        // TODO: Make previous parameters part of the signature for Objective-C
        // methods.
        if (!HadObjCArguments) {
          HadObjCArguments = true;
          // If we have no previous informative chunks (informative selector
          // fragments in practice), we treat any previous chunks as
          // `RequiredQualifiers` so they will be added as a prefix during the
          // completion.
          //
          // e.g. to complete `- (void)doSomething:(id)argument`:
          // - Completion name: `doSomething:`
          // - RequiredQualifiers: `- (void)`
          // - Snippet/Signature suffix: `(id)argument`
          //
          // This differs from the case when we're completing a method
          // expression with a previous informative selector fragment.
          //
          // e.g. to complete `[self doSomething:nil ^somethingElse:(id)]`:
          // - Previous Informative Chunk: `doSomething:`
          // - Completion name: `somethingElse:`
          // - Snippet/Signature suffix: `(id)`
          if (!HadInformativeChunks) {
            if (RequiredQualifiers)
              *RequiredQualifiers = std::move(*Signature);
            Snippet->clear();
          }
          Signature->clear();
        } else { // Subsequent argument, considered part of snippet/signature.
          *Signature += Chunk.Text;
          *Snippet += Chunk.Text;
        }
      }
      break;
    case CodeCompletionString::CK_Text:
      *Signature += Chunk.Text;
      *Snippet += Chunk.Text;
      break;
    case CodeCompletionString::CK_Optional:
      assert(Chunk.Optional);
      // No need to create placeholders for default arguments in Snippet.
      appendOptionalChunk(*Chunk.Optional, Signature);
      break;
    case CodeCompletionString::CK_Placeholder:
      *Signature += Chunk.Text;
      ++SnippetArg;
      if (SnippetArg == CursorSnippetArg) {
        // We'd like to make $0 a placeholder too, but vscode does not support
        // this (https://github.com/microsoft/vscode/issues/152837).
        *Snippet += "$0";
      } else {
        *Snippet += "${" + std::to_string(SnippetArg) + ':';
        appendEscapeSnippet(Chunk.Text, Snippet);
        *Snippet += '}';
      }
      break;
    case CodeCompletionString::CK_Informative:
      HadInformativeChunks = true;
      // For example, the word "const" for a const method, or the name of
      // the base class for methods that are part of the base class.
      *Signature += Chunk.Text;
      // Don't put the informative chunks in the snippet.
      break;
    case CodeCompletionString::CK_ResultType:
      // This is not part of the signature.
      break;
    case CodeCompletionString::CK_CurrentParameter:
      // This should never be present while collecting completion items,
      // only while collecting overload candidates.
      llvm_unreachable("Unexpected CK_CurrentParameter while collecting "
                       "CompletionItems");
      break;
    case CodeCompletionString::CK_LeftParen:
      // We're assuming that a LeftParen in a declaration starts a function
      // call, and arguments following the parenthesis could be discarded if
      // IncludeFunctionArguments is false.
      if (!IncludeFunctionArguments &&
          ResultKind == CodeCompletionResult::RK_Declaration)
        TruncateSnippetAt.emplace(Snippet->size());
      LLVM_FALLTHROUGH;
    case CodeCompletionString::CK_RightParen:
    case CodeCompletionString::CK_LeftBracket:
    case CodeCompletionString::CK_RightBracket:
    case CodeCompletionString::CK_LeftBrace:
    case CodeCompletionString::CK_RightBrace:
    case CodeCompletionString::CK_LeftAngle:
    case CodeCompletionString::CK_RightAngle:
    case CodeCompletionString::CK_Comma:
    case CodeCompletionString::CK_Colon:
    case CodeCompletionString::CK_SemiColon:
    case CodeCompletionString::CK_Equal:
    case CodeCompletionString::CK_HorizontalSpace:
      *Signature += Chunk.Text;
      *Snippet += Chunk.Text;
      break;
    case CodeCompletionString::CK_VerticalSpace:
      *Snippet += Chunk.Text;
      // Don't even add a space to the signature.
      break;
    }
  }
  if (TruncateSnippetAt)
    *Snippet = Snippet->substr(0, *TruncateSnippetAt);
}

std::string formatDocumentation(const CodeCompletionString &CCS,
                                llvm::StringRef DocComment) {
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
  if (!DocComment.empty()) {
    if (!Result.empty()) {
      // This means we previously added annotations. Add an extra newline
      // character to make the annotations stand out.
      Result.push_back('\n');
    }
    Result += DocComment;
  }
  return Result;
}

std::string getReturnType(const CodeCompletionString &CCS) {
  for (const auto &Chunk : CCS)
    if (Chunk.Kind == CodeCompletionString::CK_ResultType)
      return Chunk.Text;
  return "";
}

} // namespace clangd
} // namespace clang
