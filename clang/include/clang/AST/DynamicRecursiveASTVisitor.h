#ifndef LLVM_CLANG_AST_DYNAMIC_RECURSIVE_AST_VISITOR_H
#define LLVM_CLANG_AST_DYNAMIC_RECURSIVE_AST_VISITOR_H

#include "clang/AST/Attr.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/AST/TypeLoc.h"

namespace clang {
class ASTContext;

/// Recursive AST visitor that supports extension via dynamic dispatch.
///
/// \see RecursiveASTVisitor
class DynamicRecursiveASTVisitor {
public:
  using DataRecursionQueue =
      SmallVectorImpl<llvm::PointerIntPair<Stmt *, 1, bool>>;

  /// Whether this visitor should recurse into template instantiations.
  bool ShouldVisitTemplateInstantiations = false;

  /// Whether this visitor should recurse into the types of TypeLocs.
  bool ShouldWalkTypesOfTypeLocs = true;

  /// Whether this visitor should recurse into implicit code, e.g.
  /// implicit constructors and destructors.
  bool ShouldVisitImplicitCode = false;

  /// Whether this visitor should recurse into lambda body
  bool ShouldVisitLambdaBody = true;

  /// Return whether this visitor should traverse post-order.
  bool ShouldTraversePostOrder = false;

protected:
  DynamicRecursiveASTVisitor() = default;

public:
  // Copying/moving a polymorphic type is a bad idea.
  DynamicRecursiveASTVisitor(DynamicRecursiveASTVisitor &&) = delete;
  DynamicRecursiveASTVisitor(const DynamicRecursiveASTVisitor &) = delete;
  DynamicRecursiveASTVisitor &operator=(DynamicRecursiveASTVisitor &&) = delete;
  DynamicRecursiveASTVisitor &
  operator=(const DynamicRecursiveASTVisitor &) = delete;

  // Declared out of line as a vtable anchor.
  virtual ~DynamicRecursiveASTVisitor();

  /// Recursively visits an entire AST, starting from the TranslationUnitDecl.
  /// \returns false if visitation was terminated early.
  virtual bool TraverseAST(ASTContext &AST);

  /// Recursively visit an attribute, by dispatching to
  /// Traverse*Attr() based on the argument's dynamic type.
  ///
  /// \returns false if the visitation was terminated early, true
  /// otherwise (including when the argument is a Null type location).
  virtual bool TraverseAttr(Attr *At);

  /// Recursively visit a constructor initializer.  This
  /// automatically dispatches to another visitor for the initializer
  /// expression, but not for the name of the initializer, so may
  /// be overridden for clients that need access to the name.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  virtual bool TraverseConstructorInitializer(CXXCtorInitializer *Init);

  /// Recursively visit a declaration, by dispatching to
  /// Traverse*Decl() based on the argument's dynamic type.
  ///
  /// \returns false if the visitation was terminated early, true
  /// otherwise (including when the argument is NULL).
  virtual bool TraverseDecl(Decl *D);

  /// Recursively visit a statement or expression, by
  /// dispatching to Traverse*() based on the argument's dynamic type.
  ///
  /// \returns false if the visitation was terminated early, true
  /// otherwise (including when the argument is nullptr).
  virtual bool TraverseStmt(Stmt *S, DataRecursionQueue *Queue = nullptr);

  /// Recursively visit a template argument and dispatch to the
  /// appropriate method for the argument type.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  // FIXME: migrate callers to TemplateArgumentLoc instead.
  virtual bool TraverseTemplateArgument(const TemplateArgument &Arg);

  /// Recursively visit a set of template arguments.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  // FIXME: take a TemplateArgumentLoc* (or TemplateArgumentListInfo) instead.
  bool
  TraverseTemplateArguments(ArrayRef<TemplateArgument> Args); // NOT virtual

  /// Recursively visit a template name and dispatch to the
  /// appropriate method.
  ///
  /// \returns false if the visitation was terminated early, true otherwise.
  virtual bool TraverseTemplateName(TemplateName Template);

  /// Recursively visit a type, by dispatching to
  /// Traverse*Type() based on the argument's getTypeClass() property.
  ///
  /// \returns false if the visitation was terminated early, true
  /// otherwise (including when the argument is a Null type).
  virtual bool TraverseType(QualType T);

  /// Recursively visit a type with location, by dispatching to
  /// Traverse*TypeLoc() based on the argument type's getTypeClass() property.
  ///
  /// \returns false if the visitation was terminated early, true
  /// otherwise (including when the argument is a Null type location).
  virtual bool TraverseTypeLoc(TypeLoc TL);

  /// Traverse a concept (requirement).
  virtual bool TraverseTypeConstraint(const TypeConstraint *C);
  virtual bool TraverseConceptRequirement(concepts::Requirement *R);
  virtual bool TraverseConceptTypeRequirement(concepts::TypeRequirement *R);
  virtual bool TraverseConceptExprRequirement(concepts::ExprRequirement *R);
  virtual bool TraverseConceptNestedRequirement(concepts::NestedRequirement *R);

  /// Visit a node.
  virtual bool VisitAttr(Attr *A) { return true; }
  virtual bool VisitDecl(Decl *D) { return true; }
  virtual bool VisitStmt(Stmt *S) { return true; }
  virtual bool VisitType(Type *T) { return true; }
  virtual bool VisitTypeLoc(TypeLoc TL) { return true; }

  /// Walk up from a node.
  virtual bool WalkUpFromDecl(Decl *D) { return VisitDecl(D); }
  virtual bool WalkUpFromStmt(Stmt *S) { return VisitStmt(S); }
  virtual bool WalkUpFromType(Type *T) { return VisitType(T); }
  virtual bool WalkUpFromTypeLoc(TypeLoc TL) { return VisitTypeLoc(TL); }

  /*// Declare Traverse*() and friends for attributes.
#define DYNAMIC_ATTR_VISITOR_DECLS
#include "clang/AST/AttrVisitor.inc"
#undef DYNAMIC_ATTR_VISITOR_DECLS*/

  // Declare Traverse*() for and friends all concrete Decl classes.
#define ABSTRACT_DECL(DECL)
#define DECL(CLASS, BASE)                                                      \
  virtual bool Traverse##CLASS##Decl(CLASS##Decl *D);                          \
  virtual bool WalkUpFrom##CLASS##Decl(CLASS##Decl *D);                        \
  virtual bool Visit##CLASS##Decl(CLASS##Decl *D) { return true; }
#include "clang/AST/DeclNodes.inc"

  // Declare Traverse*() and friends for all concrete Stmt classes.
#define ABSTRACT_STMT(STMT)
#define STMT(CLASS, PARENT)                                                    \
  virtual bool Traverse##CLASS(CLASS *S);                                      \
  virtual bool WalkUpFrom##CLASS(CLASS *S);                                    \
  virtual bool Visit##CLASS(CLASS *S) { return true; }
#include "clang/AST/StmtNodes.inc"

  // Declare Traverse*() and friends for all concrete Type classes.
#define ABSTRACT_TYPE(CLASS, BASE)
#define TYPE(CLASS, BASE)                                                      \
  virtual bool Traverse##CLASS##Type(CLASS##Type *T);                          \
  virtual bool WalkUpFrom##CLASS##Type(CLASS##Type *T);                        \
  virtual bool Visit##CLASS##Type(CLASS##Type *T) { return true; }
#include "clang/AST/TypeNodes.inc"

#define ABSTRACT_TYPELOC(CLASS, BASE)
#define TYPELOC(CLASS, BASE)                                                   \
  virtual bool Traverse##CLASS##TypeLoc(CLASS##TypeLoc TL);                    \
  virtual bool WalkUpFrom##CLASS##TypeLoc(CLASS##TypeLoc TL);                  \
  virtual bool Visit##CLASS##TypeLoc(CLASS##TypeLoc TL) { return true; }
#include "clang/AST/TypeLocNodes.def"
};
} // namespace clang

#endif // LLVM_CLANG_AST_DYNAMIC_RECURSIVE_AST_VISITOR_H
