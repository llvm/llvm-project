//===--- RawCommentList.h - Classes for processing raw comments -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_RAWCOMMENTLIST_H
#define LLVM_CLANG_AST_RAWCOMMENTLIST_H

#include "clang/Basic/CommentOptions.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/ArrayRef.h"

namespace clang {

class ASTContext;
class ASTReader;
class Decl;
class Preprocessor;

namespace comments {
  class FullComment;
} // end namespace comments

class RawComment {
public:
  enum CommentKind {
    RCK_Invalid,      ///< Invalid comment
    RCK_OrdinaryBCPL, ///< Any normal BCPL comments
    RCK_OrdinaryC,    ///< Any normal C comment
    RCK_BCPLSlash,    ///< \code /// stuff \endcode
    RCK_BCPLExcl,     ///< \code //! stuff \endcode
    RCK_JavaDoc,      ///< \code /** stuff */ \endcode
    RCK_Qt,           ///< \code /*! stuff */ \endcode, also used by HeaderDoc
    RCK_Merged        ///< Two or more documentation comments merged together
  };

  RawComment() : Kind(RCK_Invalid), IsAlmostTrailingComment(false) { }

  RawComment(const SourceManager &SourceMgr, SourceRange SR,
             bool Merged, bool ParseAllComments);

  CommentKind getKind() const LLVM_READONLY {
    return (CommentKind) Kind;
  }

  bool isInvalid() const LLVM_READONLY {
    return Kind == RCK_Invalid;
  }

  bool isMerged() const LLVM_READONLY {
    return Kind == RCK_Merged;
  }

  /// Is this comment attached to any declaration?
  bool isAttached() const LLVM_READONLY {
    return IsAttached;
  }

  void setAttached() {
    IsAttached = true;
  }

  /// Returns true if it is a comment that should be put after a member:
  /// \code ///< stuff \endcode
  /// \code //!< stuff \endcode
  /// \code /**< stuff */ \endcode
  /// \code /*!< stuff */ \endcode
  bool isTrailingComment() const LLVM_READONLY {
    assert(isDocumentation());
    return IsTrailingComment;
  }

  /// Returns true if it is a probable typo:
  /// \code //< stuff \endcode
  /// \code /*< stuff */ \endcode
  bool isAlmostTrailingComment() const LLVM_READONLY {
    return IsAlmostTrailingComment;
  }

  /// Returns true if this comment is not a documentation comment.
  bool isOrdinary() const LLVM_READONLY {
    return ((Kind == RCK_OrdinaryBCPL) || (Kind == RCK_OrdinaryC)) &&
        !ParseAllComments;
  }

  /// Returns true if this comment any kind of a documentation comment.
  bool isDocumentation() const LLVM_READONLY {
    return !isInvalid() && !isOrdinary();
  }

  /// Returns whether we are parsing all comments.
  bool isParseAllComments() const LLVM_READONLY {
    return ParseAllComments;
  }

  /// Returns raw comment text with comment markers.
  StringRef getRawText(const SourceManager &SourceMgr) const {
    if (RawTextValid)
      return RawText;

    RawText = getRawTextSlow(SourceMgr);
    RawTextValid = true;
    return RawText;
  }

  SourceRange getSourceRange() const LLVM_READONLY { return Range; }
  SourceLocation getLocStart() const LLVM_READONLY { return Range.getBegin(); }
  SourceLocation getLocEnd() const LLVM_READONLY { return Range.getEnd(); }

  const char *getBriefText(const ASTContext &Context) const {
    if (BriefTextValid)
      return BriefText;

    return extractBriefText(Context);
  }

  /// Parse the comment, assuming it is attached to decl \c D.
  comments::FullComment *parse(const ASTContext &Context,
                               const Preprocessor *PP, const Decl *D) const;

private:
  SourceRange Range;

  mutable StringRef RawText;
  mutable const char *BriefText;

  mutable bool RawTextValid : 1;   ///< True if RawText is valid
  mutable bool BriefTextValid : 1; ///< True if BriefText is valid

  unsigned Kind : 3;

  /// True if comment is attached to a declaration in ASTContext.
  bool IsAttached : 1;

  bool IsTrailingComment : 1;
  bool IsAlmostTrailingComment : 1;

  /// When true, ordinary comments starting with "//" and "/*" will be
  /// considered as documentation comments.
  bool ParseAllComments : 1;

  /// \brief Constructor for AST deserialization.
  RawComment(SourceRange SR, CommentKind K, bool IsTrailingComment,
             bool IsAlmostTrailingComment,
             bool ParseAllComments) :
    Range(SR), RawTextValid(false), BriefTextValid(false), Kind(K),
    IsAttached(false), IsTrailingComment(IsTrailingComment),
    IsAlmostTrailingComment(IsAlmostTrailingComment),
    ParseAllComments(ParseAllComments)
  { }

  StringRef getRawTextSlow(const SourceManager &SourceMgr) const;

  const char *extractBriefText(const ASTContext &Context) const;

  friend class ASTReader;
};

/// \brief Compare comments' source locations.
template<>
class BeforeThanCompare<RawComment> {
  const SourceManager &SM;

public:
  explicit BeforeThanCompare(const SourceManager &SM) : SM(SM) { }

  bool operator()(const RawComment &LHS, const RawComment &RHS) {
    return SM.isBeforeInTranslationUnit(LHS.getLocStart(), RHS.getLocStart());
  }

  bool operator()(const RawComment *LHS, const RawComment *RHS) {
    return operator()(*LHS, *RHS);
  }
};

/// \brief This class represents all comments included in the translation unit,
/// sorted in order of appearance in the translation unit.
class RawCommentList {
public:
  RawCommentList(SourceManager &SourceMgr) : SourceMgr(SourceMgr) {}

  void addComment(const RawComment &RC, llvm::BumpPtrAllocator &Allocator);

  ArrayRef<RawComment *> getComments() const {
    return Comments;
  }

private:
  SourceManager &SourceMgr;
  std::vector<RawComment *> Comments;

  void addDeserializedComments(ArrayRef<RawComment *> DeserializedComments);

  friend class ASTReader;
};

} // end namespace clang

#endif
