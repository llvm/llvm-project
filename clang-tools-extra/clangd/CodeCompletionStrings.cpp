//===--- CodeCompletionStrings.cpp -------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CodeCompletionStrings.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/RawCommentList.h"
#include "clang/Basic/SourceManager.h"
#include <utility>

using namespace llvm;
namespace clang {
namespace clangd {
namespace {

bool isInformativeQualifierChunk(CodeCompletionString::Chunk const &Chunk) {
  return Chunk.Kind == CodeCompletionString::CK_Informative &&
         StringRef(Chunk.Text).endswith("::");
}

void appendEscapeSnippet(const StringRef Text, std::string *Out) {
  for (const auto Character : Text) {
    if (Character == '$' || Character == '}' || Character == '\\')
      Out->push_back('\\');
    Out->push_back(Character);
  }
}

bool looksLikeDocComment(StringRef CommentText) {
  // We don't report comments that only contain "special" chars.
  // This avoids reporting various delimiters, like:
  //   =================
  //   -----------------
  //   *****************
  return CommentText.find_first_not_of("/*-= \t\r\n") != StringRef::npos;
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
  std::string Doc = RC->getFormattedText(Ctx.getSourceManager(), Ctx.getDiagnostics());
  return looksLikeDocComment(Doc) ? Doc : "";
}

void getSignature(const CodeCompletionString &CCS, std::string *Signature,
                  std::string *Snippet, std::string *RequiredQualifiers) {
  unsigned ArgCount = 0;
  for (const auto &Chunk : CCS) {
    // Informative qualifier chunks only clutter completion results, skip
    // them.
    if (isInformativeQualifierChunk(Chunk))
      continue;

    switch (Chunk.Kind) {
    case CodeCompletionString::CK_TypedText:
      // The typed-text chunk is the actual name. We don't record this chunk.
      // In general our string looks like <qualifiers><name><signature>.
      // So once we see the name, any text we recorded so far should be
      // reclassified as qualifiers.
      if (RequiredQualifiers)
        *RequiredQualifiers = std::move(*Signature);
      Signature->clear();
      Snippet->clear();
      break;
    case CodeCompletionString::CK_Text:
      *Signature += Chunk.Text;
      *Snippet += Chunk.Text;
      break;
    case CodeCompletionString::CK_Optional:
      break;
    case CodeCompletionString::CK_Placeholder:
      *Signature += Chunk.Text;
      ++ArgCount;
      *Snippet += "${" + std::to_string(ArgCount) + ':';
      appendEscapeSnippet(Chunk.Text, Snippet);
      *Snippet += '}';
      break;
    case CodeCompletionString::CK_Informative:
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
}

std::string formatDocumentation(const CodeCompletionString &CCS,
                                StringRef DocComment) {
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
