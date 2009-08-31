//===------- TreeTransform.h - Semantic Tree Transformation ---------------===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements a semantic tree transformation that takes a given
//  AST and rebuilds it, possibly transforming some nodes in the process.
//
//===----------------------------------------------------------------------===/
#ifndef LLVM_CLANG_SEMA_TREETRANSFORM_H
#define LLVM_CLANG_SEMA_TREETRANSFORM_H

#include "Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/Parse/Ownership.h"
#include "clang/Parse/Designator.h"
#include "clang/Lex/Preprocessor.h"
#include <algorithm>

namespace clang {
  
/// \brief A semantic tree transformation that allows one to transform one
/// abstract syntax tree into another.
///
/// A new tree transformation is defined by creating a new subclass \c X of 
/// \c TreeTransform<X> and then overriding certain operations to provide 
/// behavior specific to that transformation. For example, template 
/// instantiation is implemented as a tree transformation where the
/// transformation of TemplateTypeParmType nodes involves substituting the
/// template arguments for their corresponding template parameters; a similar
/// transformation is performed for non-type template parameters and
/// template template parameters.
///
/// This tree-transformation template uses static polymorphism to allow
/// subclasses to customize any of its operations. Thus, a subclass can 
/// override any of the transformation or rebuild operators by providing an
/// operation with the same signature as the default implementation. The
/// overridding function should not be virtual.
///
/// Semantic tree transformations are split into two stages, either of which
/// can be replaced by a subclass. The "transform" step transforms an AST node
/// or the parts of an AST node using the various transformation functions,
/// then passes the pieces on to the "rebuild" step, which constructs a new AST
/// node of the appropriate kind from the pieces. The default transformation
/// routines recursively transform the operands to composite AST nodes (e.g.,
/// the pointee type of a PointerType node) and, if any of those operand nodes
/// were changed by the transformation, invokes the rebuild operation to create
/// a new AST node.
///
/// Subclasses can customize the transformation at various levels. The 
/// most coarse-grained transformations involve replacing TransformType(),
/// TransformExpr(), TransformDecl(), TransformNestedNameSpecifier(),
/// TransformTemplateName(), or TransformTemplateArgument() with entirely
/// new implementations.
///
/// For more fine-grained transformations, subclasses can replace any of the
/// \c TransformXXX functions (where XXX is the name of an AST node, e.g.,
/// PointerType, StmtExpr) to alter the transformation. As mentioned previously,
/// replacing TransformTemplateTypeParmType() allows template instantiation
/// to substitute template arguments for their corresponding template 
/// parameters. Additionally, subclasses can override the \c RebuildXXX
/// functions to control how AST nodes are rebuilt when their operands change.
/// By default, \c TreeTransform will invoke semantic analysis to rebuild
/// AST nodes. However, certain other tree transformations (e.g, cloning) may
/// be able to use more efficient rebuild steps.
///
/// There are a handful of other functions that can be overridden, allowing one
/// to avoid traversing nodes that don't need any transformation 
/// (\c AlreadyTransformed()), force rebuilding AST nodes even when their
/// operands have not changed (\c AlwaysRebuild()), and customize the
/// default locations and entity names used for type-checking
/// (\c getBaseLocation(), \c getBaseEntity()).
template<typename Derived>
class TreeTransform {
protected:
  Sema &SemaRef;
  
public:  
  typedef Sema::OwningStmtResult OwningStmtResult;
  typedef Sema::OwningExprResult OwningExprResult;
  typedef Sema::StmtArg StmtArg;
  typedef Sema::ExprArg ExprArg;
  typedef Sema::MultiExprArg MultiExprArg;
  typedef Sema::MultiStmtArg MultiStmtArg;
  
  /// \brief Initializes a new tree transformer.
  TreeTransform(Sema &SemaRef) : SemaRef(SemaRef) { }
          
  /// \brief Retrieves a reference to the derived class.
  Derived &getDerived() { return static_cast<Derived&>(*this); }

  /// \brief Retrieves a reference to the derived class.
  const Derived &getDerived() const { 
    return static_cast<const Derived&>(*this); 
  }

  /// \brief Retrieves a reference to the semantic analysis object used for
  /// this tree transform.
  Sema &getSema() const { return SemaRef; }
  
  /// \brief Whether the transformation should always rebuild AST nodes, even
  /// if none of the children have changed.
  ///
  /// Subclasses may override this function to specify when the transformation
  /// should rebuild all AST nodes.
  bool AlwaysRebuild() { return false; }
  
  /// \brief Returns the location of the entity being transformed, if that
  /// information was not available elsewhere in the AST.
  ///
  /// By default, returns no source-location information. Subclasses can 
  /// provide an alternative implementation that provides better location
  /// information.
  SourceLocation getBaseLocation() { return SourceLocation(); }
  
  /// \brief Returns the name of the entity being transformed, if that
  /// information was not available elsewhere in the AST.
  ///
  /// By default, returns an empty name. Subclasses can provide an alternative
  /// implementation with a more precise name.
  DeclarationName getBaseEntity() { return DeclarationName(); }

  /// \brief Sets the "base" location and entity when that
  /// information is known based on another transformation.
  ///
  /// By default, the source location and entity are ignored. Subclasses can
  /// override this function to provide a customized implementation.
  void setBase(SourceLocation Loc, DeclarationName Entity) { }
  
  /// \brief RAII object that temporarily sets the base location and entity
  /// used for reporting diagnostics in types.
  class TemporaryBase {
    TreeTransform &Self;
    SourceLocation OldLocation;
    DeclarationName OldEntity;
    
  public:
    TemporaryBase(TreeTransform &Self, SourceLocation Location,
                  DeclarationName Entity) : Self(Self) 
    {
      OldLocation = Self.getDerived().getBaseLocation();
      OldEntity = Self.getDerived().getBaseEntity();
      Self.getDerived().setBase(Location, Entity);
    }
    
    ~TemporaryBase() {
      Self.getDerived().setBase(OldLocation, OldEntity);
    }
  };
  
  /// \brief Determine whether the given type \p T has already been 
  /// transformed.
  ///
  /// Subclasses can provide an alternative implementation of this routine
  /// to short-circuit evaluation when it is known that a given type will 
  /// not change. For example, template instantiation need not traverse
  /// non-dependent types.
  bool AlreadyTransformed(QualType T) {
    return T.isNull();
  }

  /// \brief Transforms the given type into another type.
  ///
  /// By default, this routine transforms a type by delegating to the
  /// appropriate TransformXXXType to build a new type, then applying 
  /// the qualifiers on \p T to the resulting type with AddTypeQualifiers. 
  /// Subclasses may override this function (to take over all type 
  /// transformations), some set of the TransformXXXType functions, or
  /// the AddTypeQualifiers function to alter the transformation.
  ///
  /// \returns the transformed type.
  QualType TransformType(QualType T);
  
  /// \brief Transform the given type by adding the given set of qualifiers
  /// and returning the result.
  ///
  /// FIXME: By default, this routine adds type qualifiers only to types that
  /// can have qualifiers, and silently suppresses those qualifiers that are
  /// not permitted (e.g., qualifiers on reference or function types). This
  /// is the right thing for template instantiation, but probably not for
  /// other clients.
  QualType AddTypeQualifiers(QualType T, unsigned CVRQualifiers);
       
  /// \brief Transform the given statement.
  ///
  /// By default, this routine transforms a statement by delegating to the 
  /// appropriate TransformXXXStmt function to transform a specific kind of
  /// statement or the TransformExpr() function to transform an expression.
  /// Subclasses may override this function to transform statements using some
  /// other mechanism.
  ///
  /// \returns the transformed statement.
  OwningStmtResult TransformStmt(Stmt *S);
  
  /// \brief Transform the given expression.
  ///
  /// By default, this routine transforms an expression by delegating to the
  /// appropriate TransformXXXExpr function to build a new expression.
  /// Subclasses may override this function to transform expressions using some
  /// other mechanism.
  ///
  /// \returns the transformed expression.
  OwningExprResult TransformExpr(Expr *E) {
    return getDerived().TransformExpr(E, /*isAddressOfOperand=*/false);
  }

  /// \brief Transform the given expression.
  ///
  /// By default, this routine transforms an expression by delegating to the
  /// appropriate TransformXXXExpr function to build a new expression.
  /// Subclasses may override this function to transform expressions using some
  /// other mechanism.
  ///
  /// \returns the transformed expression.
  OwningExprResult TransformExpr(Expr *E, bool isAddressOfOperand);
  
  /// \brief Transform the given declaration, which is referenced from a type
  /// or expression.
  ///
  /// By default, acts as the identity function on declarations. Subclasses
  /// may override this function to provide alternate behavior.
  Decl *TransformDecl(Decl *D) { return D; }

  /// \brief Transform the definition of the given declaration.
  ///
  /// By default, invokes TransformDecl() to transform the declaration. 
  /// Subclasses may override this function to provide alternate behavior.
  Decl *TransformDefinition(Decl *D) { return getDerived().TransformDecl(D); }
  
  /// \brief Transform the given nested-name-specifier.
  ///
  /// By default, transforms all of the types and declarations within the 
  /// nested-name-specifier. Subclasses may override this function to provide
  /// alternate behavior.
  NestedNameSpecifier *TransformNestedNameSpecifier(NestedNameSpecifier *NNS,
                                                    SourceRange Range);
  
  /// \brief Transform the given template name.
  /// 
  /// By default, transforms the template name by transforming the declarations
  /// and nested-name-specifiers that occur within the template name. 
  /// Subclasses may override this function to provide alternate behavior.
  TemplateName TransformTemplateName(TemplateName Name);
  
  /// \brief Transform the given template argument.
  ///
  /// By default, this operation transforms the type, expression, or 
  /// declaration stored within the template argument and constructs a 
  /// new template argument from the transformed result. Subclasses may
  /// override this function to provide alternate behavior.
  TemplateArgument TransformTemplateArgument(const TemplateArgument &Arg);
  
#define ABSTRACT_TYPE(CLASS, PARENT)
#define TYPE(CLASS, PARENT)                                   \
  QualType Transform##CLASS##Type(const CLASS##Type *T);
#include "clang/AST/TypeNodes.def"      

  OwningStmtResult TransformCompoundStmt(CompoundStmt *S, bool IsStmtExpr);
                                         
#define STMT(Node, Parent)                        \
  OwningStmtResult Transform##Node(Node *S);
#define EXPR(Node, Parent)                        \
  OwningExprResult Transform##Node(Node *E);
#define ABSTRACT_EXPR(Node, Parent)
#include "clang/AST/StmtNodes.def"
  
  /// \brief Build a new pointer type given its pointee type.
  ///
  /// By default, performs semantic analysis when building the pointer type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildPointerType(QualType PointeeType);

  /// \brief Build a new block pointer type given its pointee type.
  ///
  /// By default, performs semantic analysis when building the block pointer 
  /// type. Subclasses may override this routine to provide different behavior.
  QualType RebuildBlockPointerType(QualType PointeeType);

  /// \brief Build a new lvalue reference type given the type it references.
  ///
  /// By default, performs semantic analysis when building the lvalue reference
  /// type. Subclasses may override this routine to provide different behavior.
  QualType RebuildLValueReferenceType(QualType ReferentType);

  /// \brief Build a new rvalue reference type given the type it references.
  ///
  /// By default, performs semantic analysis when building the rvalue reference
  /// type. Subclasses may override this routine to provide different behavior.
  QualType RebuildRValueReferenceType(QualType ReferentType);
  
  /// \brief Build a new member pointer type given the pointee type and the
  /// class type it refers into.
  ///
  /// By default, performs semantic analysis when building the member pointer
  /// type. Subclasses may override this routine to provide different behavior.
  QualType RebuildMemberPointerType(QualType PointeeType, QualType ClassType);
  
  /// \brief Build a new array type given the element type, size
  /// modifier, size of the array (if known), size expression, and index type
  /// qualifiers.
  ///
  /// By default, performs semantic analysis when building the array type.
  /// Subclasses may override this routine to provide different behavior.
  /// Also by default, all of the other Rebuild*Array 
  QualType RebuildArrayType(QualType ElementType,
                            ArrayType::ArraySizeModifier SizeMod,
                            const llvm::APInt *Size,
                            Expr *SizeExpr,
                            unsigned IndexTypeQuals,
                            SourceRange BracketsRange);
  
  /// \brief Build a new constant array type given the element type, size
  /// modifier, (known) size of the array, and index type qualifiers.
  ///
  /// By default, performs semantic analysis when building the array type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildConstantArrayType(QualType ElementType, 
                                    ArrayType::ArraySizeModifier SizeMod,
                                    const llvm::APInt &Size,
                                    unsigned IndexTypeQuals);

  /// \brief Build a new constant array type given the element type, size
  /// modifier, (known) size of the array, size expression, and index type 
  /// qualifiers.
  ///
  /// By default, performs semantic analysis when building the array type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildConstantArrayWithExprType(QualType ElementType, 
                                            ArrayType::ArraySizeModifier SizeMod,
                                            const llvm::APInt &Size,
                                            Expr *SizeExpr,
                                            unsigned IndexTypeQuals,
                                            SourceRange BracketsRange);

  /// \brief Build a new constant array type given the element type, size
  /// modifier, (known) size of the array, and index type qualifiers.
  ///
  /// By default, performs semantic analysis when building the array type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildConstantArrayWithoutExprType(QualType ElementType, 
                                               ArrayType::ArraySizeModifier SizeMod,
                                               const llvm::APInt &Size,
                                               unsigned IndexTypeQuals);

  /// \brief Build a new incomplete array type given the element type, size
  /// modifier, and index type qualifiers.
  ///
  /// By default, performs semantic analysis when building the array type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildIncompleteArrayType(QualType ElementType, 
                                      ArrayType::ArraySizeModifier SizeMod,
                                      unsigned IndexTypeQuals);

  /// \brief Build a new variable-length array type given the element type, 
  /// size modifier, size expression, and index type qualifiers.
  ///
  /// By default, performs semantic analysis when building the array type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildVariableArrayType(QualType ElementType, 
                                    ArrayType::ArraySizeModifier SizeMod,
                                    ExprArg SizeExpr,
                                    unsigned IndexTypeQuals,
                                    SourceRange BracketsRange);

  /// \brief Build a new dependent-sized array type given the element type, 
  /// size modifier, size expression, and index type qualifiers.
  ///
  /// By default, performs semantic analysis when building the array type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildDependentSizedArrayType(QualType ElementType, 
                                          ArrayType::ArraySizeModifier SizeMod,
                                          ExprArg SizeExpr,
                                          unsigned IndexTypeQuals,
                                          SourceRange BracketsRange);

  /// \brief Build a new vector type given the element type and
  /// number of elements.
  ///
  /// By default, performs semantic analysis when building the vector type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildVectorType(QualType ElementType, unsigned NumElements);
  
  /// \brief Build a new extended vector type given the element type and
  /// number of elements.
  ///
  /// By default, performs semantic analysis when building the vector type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildExtVectorType(QualType ElementType, unsigned NumElements,
                                SourceLocation AttributeLoc);
  
  /// \brief Build a new potentially dependently-sized extended vector type 
  /// given the element type and number of elements.
  ///
  /// By default, performs semantic analysis when building the vector type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildDependentSizedExtVectorType(QualType ElementType, 
                                              ExprArg SizeExpr,
                                              SourceLocation AttributeLoc);
    
  /// \brief Build a new function type.
  ///
  /// By default, performs semantic analysis when building the function type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildFunctionProtoType(QualType T,
                                    QualType *ParamTypes, 
                                    unsigned NumParamTypes,
                                    bool Variadic, unsigned Quals);
    
  /// \brief Build a new typedef type.
  QualType RebuildTypedefType(TypedefDecl *Typedef) {
    return SemaRef.Context.getTypeDeclType(Typedef);
  }

  /// \brief Build a new class/struct/union type.
  QualType RebuildRecordType(RecordDecl *Record) {
    return SemaRef.Context.getTypeDeclType(Record);
  }

  /// \brief Build a new Enum type.
  QualType RebuildEnumType(EnumDecl *Enum) {
    return SemaRef.Context.getTypeDeclType(Enum);
  }
  
  /// \brief Build a new typeof(expr) type. 
  ///
  /// By default, performs semantic analysis when building the typeof type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildTypeOfExprType(ExprArg Underlying);

  /// \brief Build a new typeof(type) type. 
  ///
  /// By default, builds a new TypeOfType with the given underlying type.
  QualType RebuildTypeOfType(QualType Underlying);

  /// \brief Build a new C++0x decltype type. 
  ///
  /// By default, performs semantic analysis when building the decltype type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildDecltypeType(ExprArg Underlying);
  
  /// \brief Build a new template specialization type.
  ///
  /// By default, performs semantic analysis when building the template
  /// specialization type. Subclasses may override this routine to provide
  /// different behavior.
  QualType RebuildTemplateSpecializationType(TemplateName Template,
                                             const TemplateArgument *Args,
                                             unsigned NumArgs);
  
  /// \brief Build a new qualified name type.
  ///
  /// By default, builds a new QualifiedNameType type from the 
  /// nested-name-specifier and the named type. Subclasses may override 
  /// this routine to provide different behavior.
  QualType RebuildQualifiedNameType(NestedNameSpecifier *NNS, QualType Named) {
    return SemaRef.Context.getQualifiedNameType(NNS, Named);
  }  

  /// \brief Build a new typename type that refers to a template-id.
  ///
  /// By default, builds a new TypenameType type from the nested-name-specifier
  /// and the given type. Subclasses may override this routine to provide 
  /// different behavior.
  QualType RebuildTypenameType(NestedNameSpecifier *NNS, QualType T) {
    if (NNS->isDependent())
      return SemaRef.Context.getTypenameType(NNS, 
                                          cast<TemplateSpecializationType>(T));
    
    return SemaRef.Context.getQualifiedNameType(NNS, T);
  }  

  /// \brief Build a new typename type that refers to an identifier.
  ///
  /// By default, performs semantic analysis when building the typename type
  /// (or qualified name type). Subclasses may override this routine to provide 
  /// different behavior.
  QualType RebuildTypenameType(NestedNameSpecifier *NNS, 
                               const IdentifierInfo *Id) {
    return SemaRef.CheckTypenameType(NNS, *Id,
                                  SourceRange(getDerived().getBaseLocation()));
  }
  
  /// \brief Build a new nested-name-specifier given the prefix and an
  /// identifier that names the next step in the nested-name-specifier.
  ///
  /// By default, performs semantic analysis when building the new
  /// nested-name-specifier. Subclasses may override this routine to provide
  /// different behavior.
  NestedNameSpecifier *RebuildNestedNameSpecifier(NestedNameSpecifier *Prefix,
                                                  SourceRange Range,
                                                  IdentifierInfo &II);

  /// \brief Build a new nested-name-specifier given the prefix and the
  /// namespace named in the next step in the nested-name-specifier.
  ///
  /// By default, performs semantic analysis when building the new
  /// nested-name-specifier. Subclasses may override this routine to provide
  /// different behavior.
  NestedNameSpecifier *RebuildNestedNameSpecifier(NestedNameSpecifier *Prefix,
                                                  SourceRange Range,
                                                  NamespaceDecl *NS);

  /// \brief Build a new nested-name-specifier given the prefix and the
  /// type named in the next step in the nested-name-specifier.
  ///
  /// By default, performs semantic analysis when building the new
  /// nested-name-specifier. Subclasses may override this routine to provide
  /// different behavior.
  NestedNameSpecifier *RebuildNestedNameSpecifier(NestedNameSpecifier *Prefix,
                                                  SourceRange Range,
                                                  bool TemplateKW,
                                                  QualType T);

  /// \brief Build a new template name given a nested name specifier, a flag
  /// indicating whether the "template" keyword was provided, and the template
  /// that the template name refers to.
  ///
  /// By default, builds the new template name directly. Subclasses may override
  /// this routine to provide different behavior.
  TemplateName RebuildTemplateName(NestedNameSpecifier *Qualifier,
                                   bool TemplateKW,
                                   TemplateDecl *Template);

  /// \brief Build a new template name given a nested name specifier, a flag
  /// indicating whether the "template" keyword was provided, and a set of
  /// overloaded function templates.
  ///
  /// By default, builds the new template name directly. Subclasses may override
  /// this routine to provide different behavior.
  TemplateName RebuildTemplateName(NestedNameSpecifier *Qualifier,
                                   bool TemplateKW,
                                   OverloadedFunctionDecl *Ovl);
  
  /// \brief Build a new template name given a nested name specifier and the
  /// name that is referred to as a template.
  ///
  /// By default, performs semantic analysis to determine whether the name can
  /// be resolved to a specific template, then builds the appropriate kind of
  /// template name. Subclasses may override this routine to provide different
  /// behavior.
  TemplateName RebuildTemplateName(NestedNameSpecifier *Qualifier,
                                   const IdentifierInfo &II);
  
  
  /// \brief Build a new compound statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildCompoundStmt(SourceLocation LBraceLoc,
                                       MultiStmtArg Statements,
                                       SourceLocation RBraceLoc,
                                       bool IsStmtExpr) {
    return getSema().ActOnCompoundStmt(LBraceLoc, RBraceLoc, move(Statements),
                                       IsStmtExpr);
  }

  /// \brief Build a new case statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildCaseStmt(SourceLocation CaseLoc,
                                   ExprArg LHS,
                                   SourceLocation EllipsisLoc,
                                   ExprArg RHS,
                                   SourceLocation ColonLoc) {
    return getSema().ActOnCaseStmt(CaseLoc, move(LHS), EllipsisLoc, move(RHS), 
                                   ColonLoc);
  }
  
  /// \brief Attach the body to a new case statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildCaseStmtBody(StmtArg S, StmtArg Body) {
    getSema().ActOnCaseStmtBody(S.get(), move(Body));
    return move(S);
  }
  
  /// \brief Build a new default statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildDefaultStmt(SourceLocation DefaultLoc, 
                                      SourceLocation ColonLoc,
                                      StmtArg SubStmt) {
    return getSema().ActOnDefaultStmt(DefaultLoc, ColonLoc, move(SubStmt), 
                                      /*CurScope=*/0);
  }
  
  /// \brief Build a new label statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildLabelStmt(SourceLocation IdentLoc, 
                                    IdentifierInfo *Id,
                                    SourceLocation ColonLoc,
                                    StmtArg SubStmt) {
    return SemaRef.ActOnLabelStmt(IdentLoc, Id, ColonLoc, move(SubStmt));
  }
  
  /// \brief Build a new "if" statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildIfStmt(SourceLocation IfLoc, Sema::FullExprArg Cond, 
                                 StmtArg Then, SourceLocation ElseLoc, 
                                 StmtArg Else) {
    return getSema().ActOnIfStmt(IfLoc, Cond, move(Then), ElseLoc, move(Else));
  }
  
  /// \brief Start building a new switch statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildSwitchStmtStart(ExprArg Cond) {
    return getSema().ActOnStartOfSwitchStmt(move(Cond));
  }
  
  /// \brief Attach the body to the switch statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildSwitchStmtBody(SourceLocation SwitchLoc, 
                                         StmtArg Switch, StmtArg Body) {
    return getSema().ActOnFinishSwitchStmt(SwitchLoc, move(Switch),
                                         move(Body));
  }

  /// \brief Build a new while statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildWhileStmt(SourceLocation WhileLoc,
                                    Sema::FullExprArg Cond,
                                    StmtArg Body) {
    return getSema().ActOnWhileStmt(WhileLoc, Cond, move(Body));
  }
  
  /// \brief Build a new do-while statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildDoStmt(SourceLocation DoLoc, StmtArg Body,
                                 SourceLocation WhileLoc,
                                 SourceLocation LParenLoc,
                                 ExprArg Cond,
                                 SourceLocation RParenLoc) {
    return getSema().ActOnDoStmt(DoLoc, move(Body), WhileLoc, LParenLoc, 
                                 move(Cond), RParenLoc);
  }

  /// \brief Build a new for statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildForStmt(SourceLocation ForLoc, 
                                  SourceLocation LParenLoc,
                                  StmtArg Init, ExprArg Cond, ExprArg Inc,
                                  SourceLocation RParenLoc, StmtArg Body) {
    return getSema().ActOnForStmt(ForLoc, LParenLoc, move(Init), move(Cond), 
                                  move(Inc), RParenLoc, move(Body));
  }
  
  /// \brief Build a new goto statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildGotoStmt(SourceLocation GotoLoc,
                                   SourceLocation LabelLoc,
                                   LabelStmt *Label) {
    return getSema().ActOnGotoStmt(GotoLoc, LabelLoc, Label->getID());
  }

  /// \brief Build a new indirect goto statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildIndirectGotoStmt(SourceLocation GotoLoc,
                                           SourceLocation StarLoc,
                                           ExprArg Target) {
    return getSema().ActOnIndirectGotoStmt(GotoLoc, StarLoc, move(Target));
  }
  
  /// \brief Build a new return statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildReturnStmt(SourceLocation ReturnLoc,
                                     ExprArg Result) {
    
    return getSema().ActOnReturnStmt(ReturnLoc, move(Result));
  }
  
  /// \brief Build a new declaration statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildDeclStmt(Decl **Decls, unsigned NumDecls,
                                   SourceLocation StartLoc, 
                                   SourceLocation EndLoc) {
    return getSema().Owned(
             new (getSema().Context) DeclStmt(
                                        DeclGroupRef::Create(getSema().Context,
                                                             Decls, NumDecls),
                                              StartLoc, EndLoc));
  }
  
  /// \brief Build a new C++ exception declaration.
  ///
  /// By default, performs semantic analysis to build the new decaration.
  /// Subclasses may override this routine to provide different behavior.
  VarDecl *RebuildExceptionDecl(VarDecl *ExceptionDecl, QualType T, 
                                DeclaratorInfo *Declarator,
                                IdentifierInfo *Name,
                                SourceLocation Loc,
                                SourceRange TypeRange) {
    return getSema().BuildExceptionDeclaration(0, T, Declarator, Name, Loc, 
                                               TypeRange);
  }

  /// \brief Build a new C++ catch statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildCXXCatchStmt(SourceLocation CatchLoc,
                                       VarDecl *ExceptionDecl,
                                       StmtArg Handler) {
    return getSema().Owned(
             new (getSema().Context) CXXCatchStmt(CatchLoc, ExceptionDecl, 
                                                  Handler.takeAs<Stmt>()));
  }
  
  /// \brief Build a new C++ try statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildCXXTryStmt(SourceLocation TryLoc,
                                     StmtArg TryBlock,
                                     MultiStmtArg Handlers) {
    return getSema().ActOnCXXTryBlock(TryLoc, move(TryBlock), move(Handlers));
  }
  
  /// \brief Build a new expression that references a declaration.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildDeclRefExpr(NamedDecl *ND, SourceLocation Loc) {
    return getSema().BuildDeclarationNameExpr(Loc, ND,
                                              /*FIXME:*/false,
                                              /*SS=*/0,
                                              /*FIXME:*/false);
  }
  
  /// \brief Build a new expression in parentheses.
  /// 
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildParenExpr(ExprArg SubExpr, SourceLocation LParen,
                                    SourceLocation RParen) {
    return getSema().ActOnParenExpr(LParen, RParen, move(SubExpr));
  }

  /// \brief Build a new unary operator expression.
  /// 
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildUnaryOperator(SourceLocation OpLoc,
                                        UnaryOperator::Opcode Opc,
                                        ExprArg SubExpr) {
    return getSema().CreateBuiltinUnaryOp(OpLoc, Opc, move(SubExpr));
  }
  
  /// \brief Build a new sizeof or alignof expression with a type argument.
  /// 
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildSizeOfAlignOf(QualType T, SourceLocation OpLoc,
                                        bool isSizeOf, SourceRange R) {
    return getSema().CreateSizeOfAlignOfExpr(T, OpLoc, isSizeOf, R);
  }

  /// \brief Build a new sizeof or alignof expression with an expression 
  /// argument.
  /// 
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildSizeOfAlignOf(ExprArg SubExpr, SourceLocation OpLoc,
                                        bool isSizeOf, SourceRange R) {
    OwningExprResult Result 
      = getSema().CreateSizeOfAlignOfExpr((Expr *)SubExpr.get(),
                                          OpLoc, isSizeOf, R);
    if (Result.isInvalid())
      return getSema().ExprError();
    
    SubExpr.release();
    return move(Result);
  }
  
  /// \brief Build a new array subscript expression.
  /// 
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildArraySubscriptExpr(ExprArg LHS, 
                                             SourceLocation LBracketLoc,
                                             ExprArg RHS,
                                             SourceLocation RBracketLoc) {
    return getSema().ActOnArraySubscriptExpr(/*Scope=*/0, move(LHS),
                                             LBracketLoc, move(RHS), 
                                             RBracketLoc);
  }

  /// \brief Build a new call expression.
  /// 
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCallExpr(ExprArg Callee, SourceLocation LParenLoc,
                                   MultiExprArg Args,
                                   SourceLocation *CommaLocs,
                                   SourceLocation RParenLoc) {
    return getSema().ActOnCallExpr(/*Scope=*/0, move(Callee), LParenLoc,
                                   move(Args), CommaLocs, RParenLoc);
  }

  /// \brief Build a new member access expression.
  /// 
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildMemberExpr(ExprArg Base, SourceLocation OpLoc,
                                     bool isArrow, SourceLocation MemberLoc,
                                     NamedDecl *Member) {
    return getSema().ActOnMemberReferenceExpr(/*Scope=*/0, move(Base), OpLoc,
                                              isArrow? tok::arrow : tok::period,
                                              MemberLoc,
                                              /*FIXME*/*Member->getIdentifier(),
                                     /*FIXME?*/Sema::DeclPtrTy::make((Decl*)0));
  }
  
  /// \brief Build a new binary operator expression.
  /// 
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildBinaryOperator(SourceLocation OpLoc,
                                         BinaryOperator::Opcode Opc,
                                         ExprArg LHS, ExprArg RHS) {
    OwningExprResult Result
      = getSema().CreateBuiltinBinOp(OpLoc, Opc, (Expr *)LHS.get(), 
                                     (Expr *)RHS.get());
    if (Result.isInvalid())
      return SemaRef.ExprError();
    
    LHS.release();
    RHS.release();
    return move(Result);
  }

  /// \brief Build a new conditional operator expression.
  /// 
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildConditionalOperator(ExprArg Cond,
                                              SourceLocation QuestionLoc,
                                              ExprArg LHS,
                                              SourceLocation ColonLoc,
                                              ExprArg RHS) {
    return getSema().ActOnConditionalOp(QuestionLoc, ColonLoc, move(Cond), 
                                        move(LHS), move(RHS));
  }

  /// \brief Build a new implicit cast expression.
  /// 
  /// By default, builds a new implicit cast without any semantic analysis.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildImplicitCastExpr(QualType T, CastExpr::CastKind Kind,
                                           ExprArg SubExpr, bool isLvalue) {
    ImplicitCastExpr *ICE
      = new (getSema().Context) ImplicitCastExpr(T, Kind, 
                                                 (Expr *)SubExpr.release(),
                                                 isLvalue);
    return getSema().Owned(ICE);
  }

  /// \brief Build a new C-style cast expression.
  /// 
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCStyleCaseExpr(SourceLocation LParenLoc,
                                         QualType ExplicitTy,
                                         SourceLocation RParenLoc,
                                         ExprArg SubExpr) {
    return getSema().ActOnCastExpr(/*Scope=*/0,
                                   LParenLoc,
                                   ExplicitTy.getAsOpaquePtr(),
                                   RParenLoc,
                                   move(SubExpr));
  }
  
  /// \brief Build a new compound literal expression.
  /// 
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCompoundLiteralExpr(SourceLocation LParenLoc,
                                              QualType T,
                                              SourceLocation RParenLoc,
                                              ExprArg Init) {
    return getSema().ActOnCompoundLiteral(LParenLoc, T.getAsOpaquePtr(),
                                          RParenLoc, move(Init));
  }
            
  /// \brief Build a new extended vector element access expression.
  /// 
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildExtVectorElementExpr(ExprArg Base, 
                                               SourceLocation OpLoc,
                                               SourceLocation AccessorLoc,
                                               IdentifierInfo &Accessor) {
    return getSema().ActOnMemberReferenceExpr(/*Scope=*/0, move(Base), OpLoc,
                                              tok::period, AccessorLoc,
                                              Accessor,
                                     /*FIXME?*/Sema::DeclPtrTy::make((Decl*)0));
  }
  
  /// \brief Build a new initializer list expression.
  /// 
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildInitList(SourceLocation LBraceLoc,
                                   MultiExprArg Inits,
                                   SourceLocation RBraceLoc) {
    return SemaRef.ActOnInitList(LBraceLoc, move(Inits), RBraceLoc);
  }
  
  /// \brief Build a new designated initializer expression.
  /// 
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildDesignatedInitExpr(Designation &Desig,
                                             MultiExprArg ArrayExprs,
                                             SourceLocation EqualOrColonLoc,
                                             bool GNUSyntax,
                                             ExprArg Init) {
    OwningExprResult Result
      = SemaRef.ActOnDesignatedInitializer(Desig, EqualOrColonLoc, GNUSyntax,
                                           move(Init));
    if (Result.isInvalid())
      return SemaRef.ExprError();
    
    ArrayExprs.release();
    return move(Result);
  }
  
  /// \brief Build a new value-initialized expression.
  /// 
  /// By default, builds the implicit value initialization without performing
  /// any semantic analysis. Subclasses may override this routine to provide
  /// different behavior.
  OwningExprResult RebuildImplicitValueInitExpr(QualType T) {
    return SemaRef.Owned(new (SemaRef.Context) ImplicitValueInitExpr(T));
  }
  
  /// \brief Build a new \c va_arg expression.
  /// 
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildVAArgExpr(SourceLocation BuiltinLoc, ExprArg SubExpr,
                                    QualType T, SourceLocation RParenLoc) {
    return getSema().ActOnVAArg(BuiltinLoc, move(SubExpr), T.getAsOpaquePtr(), 
                                RParenLoc);
  }

  /// \brief Build a new expression list in parentheses.
  /// 
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildParenListExpr(SourceLocation LParenLoc,
                                        MultiExprArg SubExprs,
                                        SourceLocation RParenLoc) {
    return getSema().ActOnParenListExpr(LParenLoc, RParenLoc, move(SubExprs));
  }
  
  /// \brief Build a new address-of-label expression.
  /// 
  /// By default, performs semantic analysis, using the name of the label 
  /// rather than attempting to map the label statement itself.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildAddrLabelExpr(SourceLocation AmpAmpLoc,
                                        SourceLocation LabelLoc,
                                        LabelStmt *Label) {
    return getSema().ActOnAddrLabel(AmpAmpLoc, LabelLoc, Label->getID());
  }
  
  /// \brief Build a new GNU statement expression.
  /// 
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildStmtExpr(SourceLocation LParenLoc,
                                   StmtArg SubStmt,
                                   SourceLocation RParenLoc) {
    return getSema().ActOnStmtExpr(LParenLoc, move(SubStmt), RParenLoc);
  }
  
  /// \brief Build a new __builtin_types_compatible_p expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildTypesCompatibleExpr(SourceLocation BuiltinLoc,
                                              QualType T1, QualType T2,
                                              SourceLocation RParenLoc) {
    return getSema().ActOnTypesCompatibleExpr(BuiltinLoc,
                                              T1.getAsOpaquePtr(),
                                              T2.getAsOpaquePtr(),
                                              RParenLoc);
  }
  
  /// \brief Build a new __builtin_choose_expr expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildChooseExpr(SourceLocation BuiltinLoc,
                                     ExprArg Cond, ExprArg LHS, ExprArg RHS,
                                     SourceLocation RParenLoc) {
    return SemaRef.ActOnChooseExpr(BuiltinLoc,
                                   move(Cond), move(LHS), move(RHS),
                                   RParenLoc);
  }
  
  /// \brief Build a new overloaded operator call expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// The semantic analysis provides the behavior of template instantiation,
  /// copying with transformations that turn what looks like an overloaded
  /// operator call into a use of a builtin operator, performing 
  /// argument-dependent lookup, etc. Subclasses may override this routine to
  /// provide different behavior.
  OwningExprResult RebuildCXXOperatorCallExpr(OverloadedOperatorKind Op,
                                              SourceLocation OpLoc,
                                              ExprArg Callee,
                                              ExprArg First,
                                              ExprArg Second);
  
  /// \brief Build a new C++ "named" cast expression, such as static_cast or 
  /// reinterpret_cast.
  ///
  /// By default, this routine dispatches to one of the more-specific routines
  /// for a particular named case, e.g., RebuildCXXStaticCastExpr().  
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXNamedCastExpr(SourceLocation OpLoc,
                                           Stmt::StmtClass Class,
                                           SourceLocation LAngleLoc,
                                           QualType T,
                                           SourceLocation RAngleLoc,
                                           SourceLocation LParenLoc,
                                           ExprArg SubExpr,
                                           SourceLocation RParenLoc) {
    switch (Class) {
    case Stmt::CXXStaticCastExprClass:
      return getDerived().RebuildCXXStaticCastExpr(OpLoc, LAngleLoc, T, 
                                                   RAngleLoc, LParenLoc, 
                                                   move(SubExpr), RParenLoc);

    case Stmt::CXXDynamicCastExprClass:
      return getDerived().RebuildCXXDynamicCastExpr(OpLoc, LAngleLoc, T, 
                                                    RAngleLoc, LParenLoc, 
                                                    move(SubExpr), RParenLoc);
        
    case Stmt::CXXReinterpretCastExprClass:
      return getDerived().RebuildCXXReinterpretCastExpr(OpLoc, LAngleLoc, T, 
                                                        RAngleLoc, LParenLoc, 
                                                        move(SubExpr), 
                                                        RParenLoc);
        
    case Stmt::CXXConstCastExprClass:
      return getDerived().RebuildCXXConstCastExpr(OpLoc, LAngleLoc, T, 
                                                   RAngleLoc, LParenLoc, 
                                                   move(SubExpr), RParenLoc);
        
    default:
      assert(false && "Invalid C++ named cast");
      break;
    }
    
    return getSema().ExprError();
  }
  
  /// \brief Build a new C++ static_cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXStaticCastExpr(SourceLocation OpLoc,
                                            SourceLocation LAngleLoc,
                                            QualType T,
                                            SourceLocation RAngleLoc,
                                            SourceLocation LParenLoc,
                                            ExprArg SubExpr,
                                            SourceLocation RParenLoc) {
    return getSema().ActOnCXXNamedCast(OpLoc, tok::kw_static_cast,
                                       LAngleLoc, T.getAsOpaquePtr(), RAngleLoc, 
                                       LParenLoc, move(SubExpr), RParenLoc);
  }

  /// \brief Build a new C++ dynamic_cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXDynamicCastExpr(SourceLocation OpLoc,
                                             SourceLocation LAngleLoc,
                                             QualType T,
                                             SourceLocation RAngleLoc,
                                             SourceLocation LParenLoc,
                                             ExprArg SubExpr,
                                             SourceLocation RParenLoc) {
    return getSema().ActOnCXXNamedCast(OpLoc, tok::kw_dynamic_cast,
                                       LAngleLoc, T.getAsOpaquePtr(), RAngleLoc, 
                                       LParenLoc, move(SubExpr), RParenLoc);
  }

  /// \brief Build a new C++ reinterpret_cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXReinterpretCastExpr(SourceLocation OpLoc,
                                                 SourceLocation LAngleLoc,
                                                 QualType T,
                                                 SourceLocation RAngleLoc,
                                                 SourceLocation LParenLoc,
                                                 ExprArg SubExpr,
                                                 SourceLocation RParenLoc) {
    return getSema().ActOnCXXNamedCast(OpLoc, tok::kw_reinterpret_cast,
                                       LAngleLoc, T.getAsOpaquePtr(), RAngleLoc,
                                       LParenLoc, move(SubExpr), RParenLoc);
  }

  /// \brief Build a new C++ const_cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXConstCastExpr(SourceLocation OpLoc,
                                           SourceLocation LAngleLoc,
                                           QualType T,
                                           SourceLocation RAngleLoc,
                                           SourceLocation LParenLoc,
                                           ExprArg SubExpr,
                                           SourceLocation RParenLoc) {
    return getSema().ActOnCXXNamedCast(OpLoc, tok::kw_const_cast,
                                       LAngleLoc, T.getAsOpaquePtr(), RAngleLoc, 
                                       LParenLoc, move(SubExpr), RParenLoc);
  }
  
  /// \brief Build a new C++ functional-style cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXFunctionalCastExpr(SourceRange TypeRange,
                                                QualType T,
                                                SourceLocation LParenLoc,
                                                ExprArg SubExpr,
                                                SourceLocation RParenLoc) {
    void *Sub = SubExpr.takeAs<Expr>();
    return getSema().ActOnCXXTypeConstructExpr(TypeRange,
                                               T.getAsOpaquePtr(),
                                               LParenLoc,
                                         Sema::MultiExprArg(getSema(), &Sub, 1),
                                               /*CommaLocs=*/0, 
                                               RParenLoc);
  }
  
  /// \brief Build a new C++ typeid(type) expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXTypeidExpr(SourceLocation TypeidLoc,
                                        SourceLocation LParenLoc,
                                        QualType T,
                                        SourceLocation RParenLoc) {
    return getSema().ActOnCXXTypeid(TypeidLoc, LParenLoc, true, 
                                    T.getAsOpaquePtr(), RParenLoc);
  }
  
  /// \brief Build a new C++ typeid(expr) expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXTypeidExpr(SourceLocation TypeidLoc,
                                        SourceLocation LParenLoc,
                                        ExprArg Operand,
                                        SourceLocation RParenLoc) {
    OwningExprResult Result 
      = getSema().ActOnCXXTypeid(TypeidLoc, LParenLoc, false, Operand.get(),
                                 RParenLoc);
    if (Result.isInvalid())
      return getSema().ExprError();
    
    Operand.release(); // FIXME: since ActOnCXXTypeid silently took ownership
    return move(Result);
  }  
  
  /// \brief Build a new C++ "this" expression.
  ///
  /// By default, builds a new "this" expression without performing any
  /// semantic analysis. Subclasses may override this routine to provide 
  /// different behavior.
  OwningExprResult RebuildCXXThisExpr(SourceLocation ThisLoc, 
                                      QualType ThisType) {
    return getSema().Owned(
                      new (getSema().Context) CXXThisExpr(ThisLoc, ThisType));
  }

  /// \brief Build a new C++ throw expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXThrowExpr(SourceLocation ThrowLoc, ExprArg Sub) {
    return getSema().ActOnCXXThrow(ThrowLoc, move(Sub));
  }

  /// \brief Build a new C++ default-argument expression.
  ///
  /// By default, builds a new default-argument expression, which does not
  /// require any semantic analysis. Subclasses may override this routine to
  /// provide different behavior.
  OwningExprResult RebuildCXXDefaultArgExpr(ParmVarDecl *Param) {
    return getSema().Owned(CXXDefaultArgExpr::Create(getSema().Context, Param));
  }

  /// \brief Build a new C++ zero-initialization expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXZeroInitValueExpr(SourceLocation TypeStartLoc, 
                                               SourceLocation LParenLoc,
                                               QualType T,
                                               SourceLocation RParenLoc) {
    return getSema().ActOnCXXTypeConstructExpr(SourceRange(TypeStartLoc), 
                                               T.getAsOpaquePtr(), LParenLoc, 
                                               MultiExprArg(getSema(), 0, 0), 
                                               0, RParenLoc);
  }
  
  /// \brief Build a new C++ conditional declaration expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXConditionDeclExpr(SourceLocation StartLoc, 
                                               SourceLocation EqLoc,
                                               VarDecl *Var) {
    return SemaRef.Owned(new (SemaRef.Context) CXXConditionDeclExpr(StartLoc,
                                                                    EqLoc,
                                                                    Var));
  }
  
  /// \brief Build a new C++ "new" expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXNewExpr(SourceLocation StartLoc, 
                                     bool UseGlobal,
                                     SourceLocation PlacementLParen,
                                     MultiExprArg PlacementArgs,
                                     SourceLocation PlacementRParen,
                                     bool ParenTypeId, 
                                     QualType AllocType,
                                     SourceLocation TypeLoc,
                                     SourceRange TypeRange,
                                     ExprArg ArraySize,
                                     SourceLocation ConstructorLParen,
                                     MultiExprArg ConstructorArgs,
                                     SourceLocation ConstructorRParen) {
    return getSema().BuildCXXNew(StartLoc, UseGlobal, 
                                 PlacementLParen,
                                 move(PlacementArgs),
                                 PlacementRParen,
                                 ParenTypeId,
                                 AllocType,
                                 TypeLoc,
                                 TypeRange,
                                 move(ArraySize),
                                 ConstructorLParen,
                                 move(ConstructorArgs),
                                 ConstructorRParen);
  }
  
  /// \brief Build a new C++ "delete" expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXDeleteExpr(SourceLocation StartLoc,
                                        bool IsGlobalDelete,
                                        bool IsArrayForm,
                                        ExprArg Operand) {
    return getSema().ActOnCXXDelete(StartLoc, IsGlobalDelete, IsArrayForm,
                                    move(Operand));
  }
  
  /// \brief Build a new unary type trait expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildUnaryTypeTrait(UnaryTypeTrait Trait,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         QualType T,
                                         SourceLocation RParenLoc) {
    return getSema().ActOnUnaryTypeTrait(Trait, StartLoc, LParenLoc, 
                                         T.getAsOpaquePtr(), RParenLoc);
  }

  /// \brief Build a new qualified declaration reference expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildQualifiedDeclRefExpr(NestedNameSpecifier *NNS,
                                               SourceRange QualifierRange,
                                               NamedDecl *ND,
                                               SourceLocation Location,
                                               bool IsAddressOfOperand) {
    CXXScopeSpec SS;
    SS.setRange(QualifierRange);
    SS.setScopeRep(NNS);
    return getSema().ActOnDeclarationNameExpr(/*Scope=*/0, 
                                              Location,
                                              ND->getDeclName(),
                                              /*Trailing lparen=*/false,
                                              &SS,
                                              IsAddressOfOperand);
  }

  /// \brief Build a new (previously unresolved) declaration reference 
  /// expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildUnresolvedDeclRefExpr(NestedNameSpecifier *NNS,
                                                SourceRange QualifierRange,
                                                DeclarationName Name,
                                                SourceLocation Location,
                                                bool IsAddressOfOperand) {
    CXXScopeSpec SS;
    SS.setRange(QualifierRange);
    SS.setScopeRep(NNS);
    return getSema().ActOnDeclarationNameExpr(/*Scope=*/0, 
                                              Location,
                                              Name,
                                              /*Trailing lparen=*/false,
                                              &SS,
                                              IsAddressOfOperand);
  }

  /// \brief Build a new template-id expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildTemplateIdExpr(TemplateName Template,
                                         SourceLocation TemplateLoc,
                                         SourceLocation LAngleLoc,
                                         TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation RAngleLoc) {
    return getSema().BuildTemplateIdExpr(Template, TemplateLoc,
                                         LAngleLoc,
                                         TemplateArgs, NumTemplateArgs,
                                         RAngleLoc);
  }

  /// \brief Build a new object-construction expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXConstructExpr(QualType T,
                                           CXXConstructorDecl *Constructor,
                                           bool IsElidable,
                                           MultiExprArg Args) {
    unsigned NumArgs = Args.size();
    Expr **ArgsExprs = (Expr **)Args.release();
    return getSema().Owned(SemaRef.BuildCXXConstructExpr(T, Constructor,
                                                         IsElidable,
                                                         ArgsExprs,
                                                         NumArgs));
  }

  /// \brief Build a new object-construction expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXTemporaryObjectExpr(SourceLocation TypeBeginLoc,
                                                 QualType T,
                                                 SourceLocation LParenLoc,
                                                 MultiExprArg Args,
                                                 SourceLocation *Commas,
                                                 SourceLocation RParenLoc) {
    return getSema().ActOnCXXTypeConstructExpr(SourceRange(TypeBeginLoc),
                                               T.getAsOpaquePtr(),
                                               LParenLoc,
                                               move(Args),
                                               Commas,
                                               RParenLoc);
  }

  /// \brief Build a new object-construction expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXUnresolvedConstructExpr(SourceLocation TypeBeginLoc,
                                                     QualType T,
                                                     SourceLocation LParenLoc,
                                                     MultiExprArg Args,
                                                     SourceLocation *Commas,
                                                     SourceLocation RParenLoc) {
    return getSema().ActOnCXXTypeConstructExpr(SourceRange(TypeBeginLoc,
                                                           /*FIXME*/LParenLoc),
                                               T.getAsOpaquePtr(),
                                               LParenLoc,
                                               move(Args),
                                               Commas,
                                               RParenLoc);
  }
  
  /// \brief Build a new member reference expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXUnresolvedMemberExpr(ExprArg BaseE,
                                                  bool IsArrow,
                                                  SourceLocation OperatorLoc,
                                                  DeclarationName Name,
                                                  SourceLocation MemberLoc) {
    OwningExprResult Base = move(BaseE);
    tok::TokenKind OpKind = IsArrow? tok::arrow : tok::period;
    CXXScopeSpec SS;
    Base = SemaRef.ActOnCXXEnterMemberScope(0, SS, move(Base), OpKind);
    if (Base.isInvalid())
      return SemaRef.ExprError();
    
    assert(Name.getAsIdentifierInfo() && 
           "Cannot transform member references with non-identifier members");
    Base = SemaRef.ActOnMemberReferenceExpr(/*Scope=*/0,
                                            move(Base), OperatorLoc, OpKind,
                                            MemberLoc, 
                                            *Name.getAsIdentifierInfo(),
                                    /*FIXME?*/Sema::DeclPtrTy::make((Decl*)0));
    SemaRef.ActOnCXXExitMemberScope(0, SS);
    return move(Base);
  }

  /// \brief Build a new Objective-C @encode expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildObjCEncodeExpr(SourceLocation AtLoc,
                                         QualType T,
                                         SourceLocation RParenLoc) {
    return SemaRef.Owned(SemaRef.BuildObjCEncodeExpression(AtLoc, T,
                                                           RParenLoc));
  }    

  /// \brief Build a new Objective-C protocol expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildObjCProtocolExpr(ObjCProtocolDecl *Protocol,
                                           SourceLocation AtLoc,
                                           SourceLocation ProtoLoc,
                                           SourceLocation LParenLoc,
                                           SourceLocation RParenLoc) {
    return SemaRef.Owned(SemaRef.ParseObjCProtocolExpression(
                                              Protocol->getIdentifier(),
                                                             AtLoc,
                                                             ProtoLoc,
                                                             LParenLoc,
                                                             RParenLoc));                                                             
  }
                                           
  /// \brief Build a new shuffle vector expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildShuffleVectorExpr(SourceLocation BuiltinLoc,
                                            MultiExprArg SubExprs,
                                            SourceLocation RParenLoc) {
    // Find the declaration for __builtin_shufflevector
    const IdentifierInfo &Name 
      = SemaRef.Context.Idents.get("__builtin_shufflevector");
    TranslationUnitDecl *TUDecl = SemaRef.Context.getTranslationUnitDecl();
    DeclContext::lookup_result Lookup = TUDecl->lookup(DeclarationName(&Name));
    assert(Lookup.first != Lookup.second && "No __builtin_shufflevector?");
    
    // Build a reference to the __builtin_shufflevector builtin
    FunctionDecl *Builtin = cast<FunctionDecl>(*Lookup.first);
    Expr *Callee 
      = new (SemaRef.Context) DeclRefExpr(Builtin, Builtin->getType(),
                                          BuiltinLoc, false, false);
    SemaRef.UsualUnaryConversions(Callee);
    
    // Build the CallExpr 
    unsigned NumSubExprs = SubExprs.size();
    Expr **Subs = (Expr **)SubExprs.release();
    CallExpr *TheCall = new (SemaRef.Context) CallExpr(SemaRef.Context, Callee,
                                                       Subs, NumSubExprs,
                                                       Builtin->getResultType(),
                                                       RParenLoc);
    OwningExprResult OwnedCall(SemaRef.Owned(TheCall));
    
    // Type-check the __builtin_shufflevector expression.
    OwningExprResult Result = SemaRef.SemaBuiltinShuffleVector(TheCall);
    if (Result.isInvalid())
      return SemaRef.ExprError();
    
    OwnedCall.release();
    return move(Result);    
  }
};

template<typename Derived>
Sema::OwningStmtResult TreeTransform<Derived>::TransformStmt(Stmt *S) {
  if (!S)
    return SemaRef.Owned(S);
  
  switch (S->getStmtClass()) {
  case Stmt::NoStmtClass: break;
      
  // Transform individual statement nodes
#define STMT(Node, Parent)                                              \
  case Stmt::Node##Class: return getDerived().Transform##Node(cast<Node>(S));
#define EXPR(Node, Parent)
#include "clang/AST/StmtNodes.def"
      
  // Transform expressions by calling TransformExpr.
#define STMT(Node, Parent)
#define EXPR(Node, Parent) case Stmt::Node##Class:
#include "clang/AST/StmtNodes.def"
    {
      Sema::OwningExprResult E = getDerived().TransformExpr(cast<Expr>(S));
      if (E.isInvalid())
        return getSema().StmtError();
      
      return getSema().Owned(E.takeAs<Stmt>());
    }
  }  
  
  return SemaRef.Owned(S->Retain());
}
  
  
template<typename Derived>
Sema::OwningExprResult TreeTransform<Derived>::TransformExpr(Expr *E,
                                                    bool isAddressOfOperand) {
  if (!E)
    return SemaRef.Owned(E);

  switch (E->getStmtClass()) {
    case Stmt::NoStmtClass: break;
#define STMT(Node, Parent) case Stmt::Node##Class: break;
#define EXPR(Node, Parent)                                              \
    case Stmt::Node##Class: return getDerived().Transform##Node(cast<Node>(E));
#include "clang/AST/StmtNodes.def"
  }  
      
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
NestedNameSpecifier *
TreeTransform<Derived>::TransformNestedNameSpecifier(NestedNameSpecifier *NNS,
                                                     SourceRange Range) {
  // Transform the prefix of this nested name specifier.
  NestedNameSpecifier *Prefix = NNS->getPrefix();
  if (Prefix) {
    Prefix = getDerived().TransformNestedNameSpecifier(Prefix, Range);
    if (!Prefix)
      return 0;
  }
  
  switch (NNS->getKind()) {
  case NestedNameSpecifier::Identifier:
    assert(Prefix && 
           "Can't have an identifier nested-name-specifier with no prefix");
    if (!getDerived().AlwaysRebuild() && Prefix == NNS->getPrefix())
      return NNS;
      
    return getDerived().RebuildNestedNameSpecifier(Prefix, Range, 
                                                   *NNS->getAsIdentifier());
      
  case NestedNameSpecifier::Namespace: {
    NamespaceDecl *NS 
      = cast_or_null<NamespaceDecl>(
                            getDerived().TransformDecl(NNS->getAsNamespace()));
    if (!getDerived().AlwaysRebuild() && 
        Prefix == NNS->getPrefix() &&
        NS == NNS->getAsNamespace())
      return NNS;
    
    return getDerived().RebuildNestedNameSpecifier(Prefix, Range, NS);
  }
      
  case NestedNameSpecifier::Global:
    // There is no meaningful transformation that one could perform on the
    // global scope.
    return NNS;
      
  case NestedNameSpecifier::TypeSpecWithTemplate:
  case NestedNameSpecifier::TypeSpec: {
    QualType T = getDerived().TransformType(QualType(NNS->getAsType(), 0));
    if (T.isNull())
      return 0;
    
    if (!getDerived().AlwaysRebuild() &&
        Prefix == NNS->getPrefix() &&
        T == QualType(NNS->getAsType(), 0))
      return NNS;
    
    return getDerived().RebuildNestedNameSpecifier(Prefix, Range, 
                  NNS->getKind() == NestedNameSpecifier::TypeSpecWithTemplate,    
                                                   T);
  }
  }
  
  // Required to silence a GCC warning
  return 0;  
}

template<typename Derived>
TemplateName 
TreeTransform<Derived>::TransformTemplateName(TemplateName Name) {
  if (QualifiedTemplateName *QTN = Name.getAsQualifiedTemplateName()) {
    NestedNameSpecifier *NNS 
      = getDerived().TransformNestedNameSpecifier(QTN->getQualifier(),
                      /*FIXME:*/SourceRange(getDerived().getBaseLocation()));
    if (!NNS)
      return TemplateName();
    
    if (TemplateDecl *Template = QTN->getTemplateDecl()) {
      TemplateDecl *TransTemplate 
        = cast_or_null<TemplateDecl>(getDerived().TransformDecl(Template));
      if (!TransTemplate)
        return TemplateName();
      
      if (!getDerived().AlwaysRebuild() &&
          NNS == QTN->getQualifier() &&
          TransTemplate == Template)
        return Name;
      
      return getDerived().RebuildTemplateName(NNS, QTN->hasTemplateKeyword(),
                                              TransTemplate);
    }
      
    OverloadedFunctionDecl *Ovl = QTN->getOverloadedFunctionDecl();
    assert(Ovl && "Not a template name or an overload set?");
    OverloadedFunctionDecl *TransOvl 
      = cast_or_null<OverloadedFunctionDecl>(getDerived().TransformDecl(Ovl));
    if (!TransOvl)
      return TemplateName();
    
    if (!getDerived().AlwaysRebuild() &&
        NNS == QTN->getQualifier() &&
        TransOvl == Ovl)
      return Name;
    
    return getDerived().RebuildTemplateName(NNS, QTN->hasTemplateKeyword(),
                                            TransOvl);
  }
  
  if (DependentTemplateName *DTN = Name.getAsDependentTemplateName()) {
    NestedNameSpecifier *NNS 
      = getDerived().TransformNestedNameSpecifier(DTN->getQualifier(),
                        /*FIXME:*/SourceRange(getDerived().getBaseLocation()));
    if (!NNS)
      return TemplateName();
    
    if (!getDerived().AlwaysRebuild() &&
        NNS == DTN->getQualifier())
      return Name;
    
    return getDerived().RebuildTemplateName(NNS, *DTN->getName());
  }
  
  if (TemplateDecl *Template = Name.getAsTemplateDecl()) {
    TemplateDecl *TransTemplate 
      = cast_or_null<TemplateDecl>(getDerived().TransformDecl(Template));
    if (!TransTemplate)
      return TemplateName();
    
    if (!getDerived().AlwaysRebuild() &&
        TransTemplate == Template)
      return Name;
    
    return TemplateName(TransTemplate);
  }
  
  OverloadedFunctionDecl *Ovl = Name.getAsOverloadedFunctionDecl();
  assert(Ovl && "Not a template name or an overload set?");
  OverloadedFunctionDecl *TransOvl 
    = cast_or_null<OverloadedFunctionDecl>(getDerived().TransformDecl(Ovl));
  if (!TransOvl)
    return TemplateName();
  
  if (!getDerived().AlwaysRebuild() &&
      TransOvl == Ovl)
    return Name;
  
  return TemplateName(TransOvl);
}

template<typename Derived>
TemplateArgument 
TreeTransform<Derived>::TransformTemplateArgument(const TemplateArgument &Arg) {
  switch (Arg.getKind()) {
  case TemplateArgument::Null:
  case TemplateArgument::Integral:
    return Arg;
      
  case TemplateArgument::Type: {
    QualType T = getDerived().TransformType(Arg.getAsType());
    if (T.isNull())
      return TemplateArgument();
    return TemplateArgument(Arg.getLocation(), T);
  }
      
  case TemplateArgument::Declaration: {
    Decl *D = getDerived().TransformDecl(Arg.getAsDecl());
    if (!D)
      return TemplateArgument();
    return TemplateArgument(Arg.getLocation(), D);
  }
      
  case TemplateArgument::Expression: {
    // Template argument expressions are not potentially evaluated.
    EnterExpressionEvaluationContext Unevaluated(getSema(), 
                                                 Action::Unevaluated);
    
    Sema::OwningExprResult E = getDerived().TransformExpr(Arg.getAsExpr());
    if (E.isInvalid())
      return TemplateArgument();
    return TemplateArgument(E.takeAs<Expr>());
  }
      
  case TemplateArgument::Pack: {
    llvm::SmallVector<TemplateArgument, 4> TransformedArgs;
    TransformedArgs.reserve(Arg.pack_size());
    for (TemplateArgument::pack_iterator A = Arg.pack_begin(), 
                                      AEnd = Arg.pack_end();
         A != AEnd; ++A) {
      TemplateArgument TA = getDerived().TransformTemplateArgument(*A);
      if (TA.isNull())
        return TA;
      
      TransformedArgs.push_back(TA);
    }
    TemplateArgument Result;
    Result.setArgumentPack(TransformedArgs.data(), TransformedArgs.size(), 
                           true);
    return Result;
  }
  }
  
  // Work around bogus GCC warning
  return TemplateArgument();
}

//===----------------------------------------------------------------------===//
// Type transformation
//===----------------------------------------------------------------------===//

template<typename Derived>
QualType TreeTransform<Derived>::TransformType(QualType T) {
  if (getDerived().AlreadyTransformed(T))
    return T;
  
  QualType Result;
  switch (T->getTypeClass()) {
#define ABSTRACT_TYPE(CLASS, PARENT) 
#define TYPE(CLASS, PARENT)                                                  \
    case Type::CLASS:                                                        \
      Result = getDerived().Transform##CLASS##Type(                          \
                                  static_cast<CLASS##Type*>(T.getTypePtr())); \
      break;
#include "clang/AST/TypeNodes.def"      
  }
  
  if (Result.isNull() || T == Result)
    return Result;
  
  return getDerived().AddTypeQualifiers(Result, T.getCVRQualifiers());
}
  
template<typename Derived>
QualType 
TreeTransform<Derived>::AddTypeQualifiers(QualType T, unsigned CVRQualifiers) {
  if (CVRQualifiers && !T->isFunctionType() && !T->isReferenceType())
    return T.getWithAdditionalQualifiers(CVRQualifiers);
  
  return T;
}

template<typename Derived> 
QualType TreeTransform<Derived>::TransformExtQualType(const ExtQualType *T) { 
  // FIXME: Implement
  return QualType(T, 0); 
}
  
template<typename Derived>
QualType TreeTransform<Derived>::TransformBuiltinType(const BuiltinType *T) { 
  // Nothing to do
  return QualType(T, 0); 
}
  
template<typename Derived> 
QualType TreeTransform<Derived>::TransformFixedWidthIntType(
                                                  const FixedWidthIntType *T) { 
  // FIXME: Implement
  return QualType(T, 0); 
}
  
template<typename Derived>
QualType TreeTransform<Derived>::TransformComplexType(const ComplexType *T) { 
  // FIXME: Implement
  return QualType(T, 0); 
}
  
template<typename Derived> 
QualType TreeTransform<Derived>::TransformPointerType(const PointerType *T) {
  QualType PointeeType = getDerived().TransformType(T->getPointeeType());
  if (PointeeType.isNull())
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      PointeeType == T->getPointeeType())
    return QualType(T, 0);

  return getDerived().RebuildPointerType(PointeeType);
}
  
template<typename Derived> 
QualType 
TreeTransform<Derived>::TransformBlockPointerType(const BlockPointerType *T) { 
  QualType PointeeType = getDerived().TransformType(T->getPointeeType());
  if (PointeeType.isNull())
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      PointeeType == T->getPointeeType())
    return QualType(T, 0);
  
  return getDerived().RebuildBlockPointerType(PointeeType);
}

template<typename Derived> 
QualType 
TreeTransform<Derived>::TransformLValueReferenceType(
                                               const LValueReferenceType *T) { 
  QualType PointeeType = getDerived().TransformType(T->getPointeeType());
  if (PointeeType.isNull())
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      PointeeType == T->getPointeeType())
    return QualType(T, 0);
  
  return getDerived().RebuildLValueReferenceType(PointeeType);
}

template<typename Derived> 
QualType 
TreeTransform<Derived>::TransformRValueReferenceType(
                                              const RValueReferenceType *T) { 
  QualType PointeeType = getDerived().TransformType(T->getPointeeType());
  if (PointeeType.isNull())
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      PointeeType == T->getPointeeType())
    return QualType(T, 0);
  
  return getDerived().RebuildRValueReferenceType(PointeeType);
}
  
template<typename Derived>
QualType 
TreeTransform<Derived>::TransformMemberPointerType(const MemberPointerType *T) { 
  QualType PointeeType = getDerived().TransformType(T->getPointeeType());
  if (PointeeType.isNull())
    return QualType();
  
  QualType ClassType = getDerived().TransformType(QualType(T->getClass(), 0));
  if (ClassType.isNull())
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      PointeeType == T->getPointeeType() &&
      ClassType == QualType(T->getClass(), 0))
    return QualType(T, 0);

  return getDerived().RebuildMemberPointerType(PointeeType, ClassType);
}

template<typename Derived> 
QualType 
TreeTransform<Derived>::TransformConstantArrayType(const ConstantArrayType *T) { 
  QualType ElementType = getDerived().TransformType(T->getElementType());
  if (ElementType.isNull())
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      ElementType == T->getElementType())
    return QualType(T, 0);
  
  return getDerived().RebuildConstantArrayType(ElementType, 
                                               T->getSizeModifier(),
                                               T->getSize(),
                                               T->getIndexTypeQualifier());
}
  
template<typename Derived>
QualType 
TreeTransform<Derived>::TransformConstantArrayWithExprType(
                                      const ConstantArrayWithExprType *T) { 
  QualType ElementType = getDerived().TransformType(T->getElementType());
  if (ElementType.isNull())
    return QualType();
  
  // Array bounds are not potentially evaluated contexts
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);
  
  Sema::OwningExprResult Size = getDerived().TransformExpr(T->getSizeExpr());
  if (Size.isInvalid())
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      ElementType == T->getElementType() &&
      Size.get() == T->getSizeExpr())
    return QualType(T, 0);
  
  return getDerived().RebuildConstantArrayWithExprType(ElementType, 
                                                       T->getSizeModifier(),
                                                       T->getSize(),
                                                       Size.takeAs<Expr>(),
                                                   T->getIndexTypeQualifier(),
                                                       T->getBracketsRange());
}
  
template<typename Derived> 
QualType 
TreeTransform<Derived>::TransformConstantArrayWithoutExprType(
                                      const ConstantArrayWithoutExprType *T) { 
  QualType ElementType = getDerived().TransformType(T->getElementType());
  if (ElementType.isNull())
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      ElementType == T->getElementType())
    return QualType(T, 0);
  
  return getDerived().RebuildConstantArrayWithoutExprType(ElementType, 
                                                       T->getSizeModifier(),
                                                       T->getSize(),
                                                    T->getIndexTypeQualifier());
}

template<typename Derived> 
QualType TreeTransform<Derived>::TransformIncompleteArrayType(
                                              const IncompleteArrayType *T) { 
  QualType ElementType = getDerived().TransformType(T->getElementType());
  if (ElementType.isNull())
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      ElementType == T->getElementType())
    return QualType(T, 0);

  return getDerived().RebuildIncompleteArrayType(ElementType,
                                                 T->getSizeModifier(),
                                                 T->getIndexTypeQualifier());
}
  
template<typename Derived>
QualType TreeTransform<Derived>::TransformVariableArrayType(
                                                  const VariableArrayType *T) { 
  QualType ElementType = getDerived().TransformType(T->getElementType());
  if (ElementType.isNull())
    return QualType();
  
  // Array bounds are not potentially evaluated contexts
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);

  Sema::OwningExprResult Size = getDerived().TransformExpr(T->getSizeExpr());
  if (Size.isInvalid())
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      ElementType == T->getElementType() &&
      Size.get() == T->getSizeExpr()) {
    Size.take();
    return QualType(T, 0);
  }
  
  return getDerived().RebuildVariableArrayType(ElementType, 
                                               T->getSizeModifier(),
                                               move(Size),
                                               T->getIndexTypeQualifier(),
                                               T->getBracketsRange());
}
  
template<typename Derived> 
QualType TreeTransform<Derived>::TransformDependentSizedArrayType(
                                          const DependentSizedArrayType *T) { 
  QualType ElementType = getDerived().TransformType(T->getElementType());
  if (ElementType.isNull())
    return QualType();
  
  // Array bounds are not potentially evaluated contexts
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);
  
  Sema::OwningExprResult Size = getDerived().TransformExpr(T->getSizeExpr());
  if (Size.isInvalid())
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      ElementType == T->getElementType() &&
      Size.get() == T->getSizeExpr()) {
    Size.take();
    return QualType(T, 0);
  }
  
  return getDerived().RebuildDependentSizedArrayType(ElementType, 
                                                     T->getSizeModifier(),
                                                     move(Size),
                                                     T->getIndexTypeQualifier(),
                                                     T->getBracketsRange());
}
  
template<typename Derived> 
QualType TreeTransform<Derived>::TransformDependentSizedExtVectorType(
                                      const DependentSizedExtVectorType *T) { 
  QualType ElementType = getDerived().TransformType(T->getElementType());
  if (ElementType.isNull())
    return QualType();
  
  // Vector sizes are not potentially evaluated contexts
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);

  Sema::OwningExprResult Size = getDerived().TransformExpr(T->getSizeExpr());
  if (Size.isInvalid())
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      ElementType == T->getElementType() &&
      Size.get() == T->getSizeExpr()) {
    Size.take();
    return QualType(T, 0);
  }
  
  return getDerived().RebuildDependentSizedExtVectorType(ElementType, 
                                                         move(Size),
                                                         T->getAttributeLoc());
}
  
template<typename Derived> 
QualType TreeTransform<Derived>::TransformVectorType(const VectorType *T) { 
  QualType ElementType = getDerived().TransformType(T->getElementType());
  if (ElementType.isNull())
    return QualType();

  if (!getDerived().AlwaysRebuild() &&
      ElementType == T->getElementType())
    return QualType(T, 0);
  
  return getDerived().RebuildVectorType(ElementType, T->getNumElements());
}
  
template<typename Derived> 
QualType 
TreeTransform<Derived>::TransformExtVectorType(const ExtVectorType *T) { 
  QualType ElementType = getDerived().TransformType(T->getElementType());
  if (ElementType.isNull())
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      ElementType == T->getElementType())
    return QualType(T, 0);
  
  return getDerived().RebuildExtVectorType(ElementType, T->getNumElements(),
                                           /*FIXME*/SourceLocation());
}

template<typename Derived> 
QualType TreeTransform<Derived>::TransformFunctionProtoType(
                                                  const FunctionProtoType *T) { 
  QualType ResultType = getDerived().TransformType(T->getResultType());
  if (ResultType.isNull())
    return QualType();
  
  llvm::SmallVector<QualType, 4> ParamTypes;
  for (FunctionProtoType::arg_type_iterator Param = T->arg_type_begin(),
                                         ParamEnd = T->arg_type_end(); 
       Param != ParamEnd; ++Param) {
    QualType P = getDerived().TransformType(*Param);
    if (P.isNull())
      return QualType();
    
    ParamTypes.push_back(P);
  }
  
  if (!getDerived().AlwaysRebuild() &&
      ResultType == T->getResultType() &&
      std::equal(T->arg_type_begin(), T->arg_type_end(), ParamTypes.begin()))
    return QualType(T, 0);
  
  return getDerived().RebuildFunctionProtoType(ResultType, ParamTypes.data(), 
                                               ParamTypes.size(), T->isVariadic(),
                                               T->getTypeQuals());
}
  
template<typename Derived>
QualType TreeTransform<Derived>::TransformFunctionNoProtoType(
                                                const FunctionNoProtoType *T) { 
  // FIXME: Implement
  return QualType(T, 0); 
}
  
template<typename Derived>
QualType TreeTransform<Derived>::TransformTypedefType(const TypedefType *T) { 
  TypedefDecl *Typedef
    = cast_or_null<TypedefDecl>(getDerived().TransformDecl(T->getDecl()));
  if (!Typedef)
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      Typedef == T->getDecl())
    return QualType(T, 0);
  
  return getDerived().RebuildTypedefType(Typedef);
}
  
template<typename Derived>
QualType TreeTransform<Derived>::TransformTypeOfExprType(
                                                    const TypeOfExprType *T) { 
  // typeof expressions are not potentially evaluated contexts
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);
  
  Sema::OwningExprResult E = getDerived().TransformExpr(T->getUnderlyingExpr());
  if (E.isInvalid())
    return QualType();

  if (!getDerived().AlwaysRebuild() &&
      E.get() == T->getUnderlyingExpr()) {
    E.take();
    return QualType(T, 0);
  }
  
  return getDerived().RebuildTypeOfExprType(move(E));
}
  
template<typename Derived> 
QualType TreeTransform<Derived>::TransformTypeOfType(const TypeOfType *T) { 
  QualType Underlying = getDerived().TransformType(T->getUnderlyingType());
  if (Underlying.isNull())
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      Underlying == T->getUnderlyingType())
    return QualType(T, 0);
  
  return getDerived().RebuildTypeOfType(Underlying);
}
  
template<typename Derived> 
QualType TreeTransform<Derived>::TransformDecltypeType(const DecltypeType *T) { 
  // decltype expressions are not potentially evaluated contexts
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);
  
  Sema::OwningExprResult E = getDerived().TransformExpr(T->getUnderlyingExpr());
  if (E.isInvalid())
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      E.get() == T->getUnderlyingExpr()) {
    E.take();
    return QualType(T, 0);
  }
  
  return getDerived().RebuildDecltypeType(move(E));
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformRecordType(const RecordType *T) { 
  RecordDecl *Record
  = cast_or_null<RecordDecl>(getDerived().TransformDecl(T->getDecl()));
  if (!Record)
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      Record == T->getDecl())
    return QualType(T, 0);
  
  return getDerived().RebuildRecordType(Record);
}
  
template<typename Derived> 
QualType TreeTransform<Derived>::TransformEnumType(const EnumType *T) { 
  EnumDecl *Enum
  = cast_or_null<EnumDecl>(getDerived().TransformDecl(T->getDecl()));
  if (!Enum)
    return QualType();
  
  if (!getDerived().AlwaysRebuild() &&
      Enum == T->getDecl())
    return QualType(T, 0);
  
  return getDerived().RebuildEnumType(Enum);
}
  
template<typename Derived>
QualType TreeTransform<Derived>::TransformTemplateTypeParmType(
                                              const TemplateTypeParmType *T) { 
  // Nothing to do
  return QualType(T, 0); 
}

template<typename Derived> 
QualType TreeTransform<Derived>::TransformTemplateSpecializationType(
                                        const TemplateSpecializationType *T) { 
  TemplateName Template 
    = getDerived().TransformTemplateName(T->getTemplateName());
  if (Template.isNull())
    return QualType();
  
  llvm::SmallVector<TemplateArgument, 4> NewTemplateArgs;
  NewTemplateArgs.reserve(T->getNumArgs());
  for (TemplateSpecializationType::iterator Arg = T->begin(), ArgEnd = T->end();
       Arg != ArgEnd; ++Arg) {
    TemplateArgument NewArg = getDerived().TransformTemplateArgument(*Arg);
    if (NewArg.isNull())
      return QualType();
    
    NewTemplateArgs.push_back(NewArg);
  }
  
  // FIXME: early abort if all of the template arguments and such are the
  // same.
  
  // FIXME: We're missing the locations of the template name, '<', and '>'.
  return getDerived().RebuildTemplateSpecializationType(Template,
                                                        NewTemplateArgs.data(),
                                                        NewTemplateArgs.size());
}
  
template<typename Derived> 
QualType TreeTransform<Derived>::TransformQualifiedNameType(
                                                  const QualifiedNameType *T) {
  NestedNameSpecifier *NNS
    = getDerived().TransformNestedNameSpecifier(T->getQualifier(),
                                                SourceRange());
  if (!NNS)
    return QualType();
  
  QualType Named = getDerived().TransformType(T->getNamedType());
  if (Named.isNull())
    return QualType();
                      
  if (!getDerived().AlwaysRebuild() &&
      NNS == T->getQualifier() &&
      Named == T->getNamedType())
    return QualType(T, 0);

  return getDerived().RebuildQualifiedNameType(NNS, Named);
}
  
template<typename Derived> 
QualType TreeTransform<Derived>::TransformTypenameType(const TypenameType *T) {
  NestedNameSpecifier *NNS
    = getDerived().TransformNestedNameSpecifier(T->getQualifier(),
                        SourceRange(/*FIXME:*/getDerived().getBaseLocation()));
  if (!NNS)
    return QualType();
  
  if (const TemplateSpecializationType *TemplateId = T->getTemplateId()) {
    QualType NewTemplateId 
      = getDerived().TransformType(QualType(TemplateId, 0));
    if (NewTemplateId.isNull())
      return QualType();
    
    if (!getDerived().AlwaysRebuild() &&
        NNS == T->getQualifier() &&
        NewTemplateId == QualType(TemplateId, 0))
      return QualType(T, 0);
    
    return getDerived().RebuildTypenameType(NNS, NewTemplateId);
  }

  return getDerived().RebuildTypenameType(NNS, T->getIdentifier());
}
  
template<typename Derived>
QualType TreeTransform<Derived>::TransformObjCInterfaceType(
                                                  const ObjCInterfaceType *T) { 
  // FIXME: Implement
  return QualType(T, 0); 
}
  
template<typename Derived> 
QualType TreeTransform<Derived>::TransformObjCObjectPointerType(
                                             const ObjCObjectPointerType *T) { 
  // FIXME: Implement
  return QualType(T, 0); 
}

//===----------------------------------------------------------------------===//
// Statement transformation
//===----------------------------------------------------------------------===//
template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformNullStmt(NullStmt *S) { 
  return SemaRef.Owned(S->Retain()); 
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformCompoundStmt(CompoundStmt *S) {
  return getDerived().TransformCompoundStmt(S, false);
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformCompoundStmt(CompoundStmt *S, 
                                              bool IsStmtExpr) {
  bool SubStmtChanged = false;
  ASTOwningVector<&ActionBase::DeleteStmt> Statements(getSema());
  for (CompoundStmt::body_iterator B = S->body_begin(), BEnd = S->body_end();
       B != BEnd; ++B) {
    OwningStmtResult Result = getDerived().TransformStmt(*B);
    if (Result.isInvalid())
      return getSema().StmtError();
    
    SubStmtChanged = SubStmtChanged || Result.get() != *B;
    Statements.push_back(Result.takeAs<Stmt>());
  }
  
  if (!getDerived().AlwaysRebuild() &&
      !SubStmtChanged)
    return SemaRef.Owned(S->Retain()); 

  return getDerived().RebuildCompoundStmt(S->getLBracLoc(),
                                          move_arg(Statements),
                                          S->getRBracLoc(),
                                          IsStmtExpr);
}
  
template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformCaseStmt(CaseStmt *S) { 
  // The case value expressions are not potentially evaluated.
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);
  
  // Transform the left-hand case value.
  OwningExprResult LHS = getDerived().TransformExpr(S->getLHS());
  if (LHS.isInvalid())
    return SemaRef.StmtError();
  
  // Transform the right-hand case value (for the GNU case-range extension).
  OwningExprResult RHS = getDerived().TransformExpr(S->getRHS());
  if (RHS.isInvalid())
    return SemaRef.StmtError();
  
  // Build the case statement.
  // Case statements are always rebuilt so that they will attached to their
  // transformed switch statement.
  OwningStmtResult Case = getDerived().RebuildCaseStmt(S->getCaseLoc(),
                                                       move(LHS),
                                                       S->getEllipsisLoc(),
                                                       move(RHS),
                                                       S->getColonLoc());
  if (Case.isInvalid())
    return SemaRef.StmtError();
  
  // Transform the statement following the case
  OwningStmtResult SubStmt = getDerived().TransformStmt(S->getSubStmt());
  if (SubStmt.isInvalid())
    return SemaRef.StmtError();
  
  // Attach the body to the case statement
  return getDerived().RebuildCaseStmtBody(move(Case), move(SubStmt));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformDefaultStmt(DefaultStmt *S) { 
  // Transform the statement following the default case
  OwningStmtResult SubStmt = getDerived().TransformStmt(S->getSubStmt());
  if (SubStmt.isInvalid())
    return SemaRef.StmtError();
  
  // Default statements are always rebuilt
  return getDerived().RebuildDefaultStmt(S->getDefaultLoc(), S->getColonLoc(),
                                         move(SubStmt));
}
  
template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformLabelStmt(LabelStmt *S) { 
  OwningStmtResult SubStmt = getDerived().TransformStmt(S->getSubStmt());
  if (SubStmt.isInvalid())
    return SemaRef.StmtError();
  
  // FIXME: Pass the real colon location in.
  SourceLocation ColonLoc = SemaRef.PP.getLocForEndOfToken(S->getIdentLoc());
  return getDerived().RebuildLabelStmt(S->getIdentLoc(), S->getID(), ColonLoc,
                                       move(SubStmt));
}
  
template<typename Derived>
Sema::OwningStmtResult 
TreeTransform<Derived>::TransformIfStmt(IfStmt *S) { 
  // Transform the condition
  OwningExprResult Cond = getDerived().TransformExpr(S->getCond());
  if (Cond.isInvalid())
    return SemaRef.StmtError();
  
  Sema::FullExprArg FullCond(getSema().FullExpr(Cond));
  
  // Transform the "then" branch.
  OwningStmtResult Then = getDerived().TransformStmt(S->getThen());
  if (Then.isInvalid())
    return SemaRef.StmtError();
  
  // Transform the "else" branch.
  OwningStmtResult Else = getDerived().TransformStmt(S->getElse());
  if (Else.isInvalid())
    return SemaRef.StmtError();
  
  if (!getDerived().AlwaysRebuild() &&
      FullCond->get() == S->getCond() &&
      Then.get() == S->getThen() &&
      Else.get() == S->getElse())
    return SemaRef.Owned(S->Retain()); 
  
  return getDerived().RebuildIfStmt(S->getIfLoc(), FullCond, move(Then),
                                    S->getElseLoc(), move(Else));  
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformSwitchStmt(SwitchStmt *S) { 
  // Transform the condition.
  OwningExprResult Cond = getDerived().TransformExpr(S->getCond());
  if (Cond.isInvalid())
    return SemaRef.StmtError();
  
  // Rebuild the switch statement.
  OwningStmtResult Switch = getDerived().RebuildSwitchStmtStart(move(Cond));
  if (Switch.isInvalid())
    return SemaRef.StmtError();
  
  // Transform the body of the switch statement.
  OwningStmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return SemaRef.StmtError();
  
  // Complete the switch statement.
  return getDerived().RebuildSwitchStmtBody(S->getSwitchLoc(), move(Switch),
                                            move(Body));
}
  
template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformWhileStmt(WhileStmt *S) { 
  // Transform the condition
  OwningExprResult Cond = getDerived().TransformExpr(S->getCond());
  if (Cond.isInvalid())
    return SemaRef.StmtError();
  
  Sema::FullExprArg FullCond(getSema().FullExpr(Cond));
  
  // Transform the body
  OwningStmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return SemaRef.StmtError();
  
  if (!getDerived().AlwaysRebuild() &&
      FullCond->get() == S->getCond() &&
      Body.get() == S->getBody())
    return SemaRef.Owned(S->Retain()); 
  
  return getDerived().RebuildWhileStmt(S->getWhileLoc(), FullCond, move(Body));
}
  
template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformDoStmt(DoStmt *S) {
  // Transform the condition
  OwningExprResult Cond = getDerived().TransformExpr(S->getCond());
  if (Cond.isInvalid())
    return SemaRef.StmtError();
  
  // Transform the body
  OwningStmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return SemaRef.StmtError();
  
  if (!getDerived().AlwaysRebuild() &&
      Cond.get() == S->getCond() &&
      Body.get() == S->getBody())
    return SemaRef.Owned(S->Retain()); 
  
  return getDerived().RebuildDoStmt(S->getDoLoc(), move(Body), S->getWhileLoc(),
                                    /*FIXME:*/S->getWhileLoc(), move(Cond),
                                    S->getRParenLoc());
}
  
template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformForStmt(ForStmt *S) { 
  // Transform the initialization statement
  OwningStmtResult Init = getDerived().TransformStmt(S->getInit());
  if (Init.isInvalid())
    return SemaRef.StmtError();
  
  // Transform the condition
  OwningExprResult Cond = getDerived().TransformExpr(S->getCond());
  if (Cond.isInvalid())
    return SemaRef.StmtError();
  
  // Transform the increment
  OwningExprResult Inc = getDerived().TransformExpr(S->getInc());
  if (Inc.isInvalid())
    return SemaRef.StmtError();
  
  // Transform the body
  OwningStmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return SemaRef.StmtError();
  
  if (!getDerived().AlwaysRebuild() &&
      Init.get() == S->getInit() &&
      Cond.get() == S->getCond() &&
      Inc.get() == S->getInc() &&
      Body.get() == S->getBody())
    return SemaRef.Owned(S->Retain()); 
  
  return getDerived().RebuildForStmt(S->getForLoc(), S->getLParenLoc(),
                                     move(Init), move(Cond), move(Inc),
                                     S->getRParenLoc(), move(Body));
}

template<typename Derived>
Sema::OwningStmtResult 
TreeTransform<Derived>::TransformGotoStmt(GotoStmt *S) { 
  // Goto statements must always be rebuilt, to resolve the label.
  return getDerived().RebuildGotoStmt(S->getGotoLoc(), S->getLabelLoc(), 
                                      S->getLabel());
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformIndirectGotoStmt(IndirectGotoStmt *S) { 
  OwningExprResult Target = getDerived().TransformExpr(S->getTarget());
  if (Target.isInvalid())
    return SemaRef.StmtError();
  
  if (!getDerived().AlwaysRebuild() &&
      Target.get() == S->getTarget())
    return SemaRef.Owned(S->Retain()); 

  return getDerived().RebuildIndirectGotoStmt(S->getGotoLoc(), S->getStarLoc(),
                                              move(Target));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformContinueStmt(ContinueStmt *S) { 
  return SemaRef.Owned(S->Retain()); 
}
  
template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformBreakStmt(BreakStmt *S) { 
  return SemaRef.Owned(S->Retain()); 
}
  
template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformReturnStmt(ReturnStmt *S) { 
  Sema::OwningExprResult Result = getDerived().TransformExpr(S->getRetValue());
  if (Result.isInvalid())
    return SemaRef.StmtError();

  // FIXME: We always rebuild the return statement because there is no way 
  // to tell whether the return type of the function has changed.
  return getDerived().RebuildReturnStmt(S->getReturnLoc(), move(Result));
}
  
template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformDeclStmt(DeclStmt *S) { 
  bool DeclChanged = false;
  llvm::SmallVector<Decl *, 4> Decls;
  for (DeclStmt::decl_iterator D = S->decl_begin(), DEnd = S->decl_end();
       D != DEnd; ++D) {
    Decl *Transformed = getDerived().TransformDefinition(*D);
    if (!Transformed)
      return SemaRef.StmtError();
        
    if (Transformed != *D)
      DeclChanged = true;
    
    Decls.push_back(Transformed);
  }
  
  if (!getDerived().AlwaysRebuild() && !DeclChanged)
    return SemaRef.Owned(S->Retain()); 
  
  return getDerived().RebuildDeclStmt(Decls.data(), Decls.size(), 
                                      S->getStartLoc(), S->getEndLoc());
}
  
template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformSwitchCase(SwitchCase *S) { 
  assert(false && "SwitchCase is abstract and cannot be transformed");
  return SemaRef.Owned(S->Retain()); 
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformAsmStmt(AsmStmt *S) {
  // FIXME: Implement!
  assert(false && "Inline assembly cannot be transformed");
  return SemaRef.Owned(S->Retain()); 
}


template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCAtTryStmt(ObjCAtTryStmt *S) { 
  // FIXME: Implement this
  assert(false && "Cannot transform an Objective-C @try statement");
  return SemaRef.Owned(S->Retain()); 
}
  
template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCAtCatchStmt(ObjCAtCatchStmt *S) { 
  // FIXME: Implement this
  assert(false && "Cannot transform an Objective-C @catch statement");
  return SemaRef.Owned(S->Retain());
}
  
template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCAtFinallyStmt(ObjCAtFinallyStmt *S) { 
  // FIXME: Implement this
  assert(false && "Cannot transform an Objective-C @finally statement");
  return SemaRef.Owned(S->Retain()); 
}
  
template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCAtThrowStmt(ObjCAtThrowStmt *S) { 
  // FIXME: Implement this
  assert(false && "Cannot transform an Objective-C @throw statement");
  return SemaRef.Owned(S->Retain()); 
}
  
template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCAtSynchronizedStmt(
                                                  ObjCAtSynchronizedStmt *S) { 
  // FIXME: Implement this
  assert(false && "Cannot transform an Objective-C @synchronized statement");
  return SemaRef.Owned(S->Retain()); 
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCForCollectionStmt(
                                                  ObjCForCollectionStmt *S) { 
  // FIXME: Implement this
  assert(false && "Cannot transform an Objective-C for-each statement");
  return SemaRef.Owned(S->Retain()); 
}


template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformCXXCatchStmt(CXXCatchStmt *S) {
  // Transform the exception declaration, if any.
  VarDecl *Var = 0;
  if (S->getExceptionDecl()) {
    VarDecl *ExceptionDecl = S->getExceptionDecl();
    TemporaryBase Rebase(*this, ExceptionDecl->getLocation(),
                         ExceptionDecl->getDeclName());

    QualType T = getDerived().TransformType(ExceptionDecl->getType());
    if (T.isNull())
      return SemaRef.StmtError();
    
    Var = getDerived().RebuildExceptionDecl(ExceptionDecl,
                                            T,
                                            ExceptionDecl->getDeclaratorInfo(),
                                            ExceptionDecl->getIdentifier(),
                                            ExceptionDecl->getLocation(),
                                            /*FIXME: Inaccurate*/
                                    SourceRange(ExceptionDecl->getLocation()));
    if (!Var || Var->isInvalidDecl()) {
      if (Var)
        Var->Destroy(SemaRef.Context);
      return SemaRef.StmtError();
    }
  }
  
  // Transform the actual exception handler.
  OwningStmtResult Handler = getDerived().TransformStmt(S->getHandlerBlock());
  if (Handler.isInvalid()) {
    if (Var)
      Var->Destroy(SemaRef.Context);
    return SemaRef.StmtError();
  }
  
  if (!getDerived().AlwaysRebuild() &&
      !Var &&
      Handler.get() == S->getHandlerBlock())
    return SemaRef.Owned(S->Retain()); 

  return getDerived().RebuildCXXCatchStmt(S->getCatchLoc(),
                                          Var,
                                          move(Handler));
}
  
template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformCXXTryStmt(CXXTryStmt *S) {
  // Transform the try block itself.
  OwningStmtResult TryBlock 
    = getDerived().TransformCompoundStmt(S->getTryBlock());
  if (TryBlock.isInvalid())
    return SemaRef.StmtError();
  
  // Transform the handlers.
  bool HandlerChanged = false;
  ASTOwningVector<&ActionBase::DeleteStmt> Handlers(SemaRef);
  for (unsigned I = 0, N = S->getNumHandlers(); I != N; ++I) {
    OwningStmtResult Handler 
      = getDerived().TransformCXXCatchStmt(S->getHandler(I));
    if (Handler.isInvalid())
      return SemaRef.StmtError();
    
    HandlerChanged = HandlerChanged || Handler.get() != S->getHandler(I);
    Handlers.push_back(Handler.takeAs<Stmt>());
  }
  
  if (!getDerived().AlwaysRebuild() &&
      TryBlock.get() == S->getTryBlock() &&
      !HandlerChanged)
    return SemaRef.Owned(S->Retain()); 

  return getDerived().RebuildCXXTryStmt(S->getTryLoc(), move(TryBlock),
                                        move_arg(Handlers));  
}
  
//===----------------------------------------------------------------------===//
// Expression transformation
//===----------------------------------------------------------------------===//
template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformPredefinedExpr(PredefinedExpr *E) { 
  return SemaRef.Owned(E->Retain()); 
}
  
template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformDeclRefExpr(DeclRefExpr *E) { 
  NamedDecl *ND 
    = dyn_cast_or_null<NamedDecl>(getDerived().TransformDecl(E->getDecl()));
  if (!ND)
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() && ND == E->getDecl())
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildDeclRefExpr(ND, E->getLocation());
}
  
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformIntegerLiteral(IntegerLiteral *E) { 
  return SemaRef.Owned(E->Retain()); 
}
  
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformFloatingLiteral(FloatingLiteral *E) { 
  return SemaRef.Owned(E->Retain()); 
}
  
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformImaginaryLiteral(ImaginaryLiteral *E) { 
  return SemaRef.Owned(E->Retain()); 
}
  
template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformStringLiteral(StringLiteral *E) { 
  return SemaRef.Owned(E->Retain()); 
}
  
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformCharacterLiteral(CharacterLiteral *E) { 
  return SemaRef.Owned(E->Retain()); 
}
  
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformParenExpr(ParenExpr *E) { 
  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() && SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildParenExpr(move(SubExpr), E->getLParen(), 
                                       E->getRParen());
}

template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformUnaryOperator(UnaryOperator *E) { 
  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() && SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildUnaryOperator(E->getOperatorLoc(),
                                           E->getOpcode(),
                                           move(SubExpr));
}
  
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformSizeOfAlignOfExpr(SizeOfAlignOfExpr *E) { 
  if (E->isArgumentType()) {
    QualType T = getDerived().TransformType(E->getArgumentType());
    if (T.isNull())
      return SemaRef.ExprError();
    
    if (!getDerived().AlwaysRebuild() && T == E->getArgumentType())
      return SemaRef.Owned(E->Retain());
    
    return getDerived().RebuildSizeOfAlignOf(T, E->getOperatorLoc(), 
                                             E->isSizeOf(), 
                                             E->getSourceRange());
  }
  
  Sema::OwningExprResult SubExpr(SemaRef);
  {   
    // C++0x [expr.sizeof]p1:
    //   The operand is either an expression, which is an unevaluated operand
    //   [...]
    EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);
    
    SubExpr = getDerived().TransformExpr(E->getArgumentExpr());
    if (SubExpr.isInvalid())
      return SemaRef.ExprError();
    
    if (!getDerived().AlwaysRebuild() && SubExpr.get() == E->getArgumentExpr())
      return SemaRef.Owned(E->Retain());
  }
  
  return getDerived().RebuildSizeOfAlignOf(move(SubExpr), E->getOperatorLoc(),
                                           E->isSizeOf(),
                                           E->getSourceRange());
}
  
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformArraySubscriptExpr(ArraySubscriptExpr *E) {
  OwningExprResult LHS = getDerived().TransformExpr(E->getLHS());
  if (LHS.isInvalid())
    return SemaRef.ExprError();
  
  OwningExprResult RHS = getDerived().TransformExpr(E->getRHS());
  if (RHS.isInvalid())
    return SemaRef.ExprError();
  
  
  if (!getDerived().AlwaysRebuild() &&
      LHS.get() == E->getLHS() &&
      RHS.get() == E->getRHS())
    return SemaRef.Owned(E->Retain());
  
  return getDerived().RebuildArraySubscriptExpr(move(LHS),
                                           /*FIXME:*/E->getLHS()->getLocStart(),
                                                move(RHS),
                                                E->getRBracketLoc());
}
  
template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformCallExpr(CallExpr *E) {
  // Transform the callee.
  OwningExprResult Callee = getDerived().TransformExpr(E->getCallee());
  if (Callee.isInvalid())
    return SemaRef.ExprError();

  // Transform arguments.
  bool ArgChanged = false;
  ASTOwningVector<&ActionBase::DeleteExpr> Args(SemaRef);
  llvm::SmallVector<SourceLocation, 4> FakeCommaLocs;
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I) {
    OwningExprResult Arg = getDerived().TransformExpr(E->getArg(I));
    if (Arg.isInvalid())
      return SemaRef.ExprError();
    
    // FIXME: Wrong source location information for the ','.
    FakeCommaLocs.push_back(
       SemaRef.PP.getLocForEndOfToken(E->getArg(I)->getSourceRange().getEnd()));
    
    ArgChanged = ArgChanged || Arg.get() != E->getArg(I);   
    Args.push_back(Arg.takeAs<Expr>());
  }
  
  if (!getDerived().AlwaysRebuild() &&
      Callee.get() == E->getCallee() &&
      !ArgChanged)
    return SemaRef.Owned(E->Retain());
  
  // FIXME: Wrong source location information for the '('.
  SourceLocation FakeLParenLoc 
    = ((Expr *)Callee.get())->getSourceRange().getBegin();
  return getDerived().RebuildCallExpr(move(Callee), FakeLParenLoc,
                                      move_arg(Args),
                                      FakeCommaLocs.data(),
                                      E->getRParenLoc());
}
  
template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformMemberExpr(MemberExpr *E) { 
  OwningExprResult Base = getDerived().TransformExpr(E->getBase());
  if (Base.isInvalid())
    return SemaRef.ExprError();
  
  NamedDecl *Member 
    = cast_or_null<NamedDecl>(getDerived().TransformDecl(E->getMemberDecl()));
  if (!Member)
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      Base.get() == E->getBase() &&
      Member == E->getMemberDecl())
    return SemaRef.Owned(E->Retain()); 

  // FIXME: Bogus source location for the operator
  SourceLocation FakeOperatorLoc
    = SemaRef.PP.getLocForEndOfToken(E->getBase()->getSourceRange().getEnd());

  return getDerived().RebuildMemberExpr(move(Base), FakeOperatorLoc,
                                        E->isArrow(),
                                        E->getMemberLoc(),
                                        Member);                                        
}
  
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformCastExpr(CastExpr *E) { 
  assert(false && "Cannot transform abstract class");
  return SemaRef.Owned(E->Retain()); 
}
  
template<typename Derived> 
Sema::OwningExprResult
TreeTransform<Derived>::TransformBinaryOperator(BinaryOperator *E) { 
  OwningExprResult LHS = getDerived().TransformExpr(E->getLHS());
  if (LHS.isInvalid())
    return SemaRef.ExprError();
  
  OwningExprResult RHS = getDerived().TransformExpr(E->getRHS());
  if (RHS.isInvalid())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      LHS.get() == E->getLHS() &&
      RHS.get() == E->getRHS())
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildBinaryOperator(E->getOperatorLoc(), E->getOpcode(),
                                            move(LHS), move(RHS));
}

template<typename Derived> 
Sema::OwningExprResult
TreeTransform<Derived>::TransformCompoundAssignOperator(
                                                  CompoundAssignOperator *E) {
  return getDerived().TransformBinaryOperator(E);
}
  
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformConditionalOperator(ConditionalOperator *E) { 
  OwningExprResult Cond = getDerived().TransformExpr(E->getCond());
  if (Cond.isInvalid())
    return SemaRef.ExprError();
  
  OwningExprResult LHS = getDerived().TransformExpr(E->getLHS());
  if (LHS.isInvalid())
    return SemaRef.ExprError();
  
  OwningExprResult RHS = getDerived().TransformExpr(E->getRHS());
  if (RHS.isInvalid())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      Cond.get() == E->getCond() &&
      LHS.get() == E->getLHS() &&
      RHS.get() == E->getRHS())
    return SemaRef.Owned(E->Retain());
  
  // FIXM: ? and : locations are broken.
  SourceLocation FakeQuestionLoc = E->getCond()->getLocEnd();
  SourceLocation FakeColonLoc = E->getFalseExpr()->getLocStart();
  return getDerived().RebuildConditionalOperator(move(Cond), 
                                                 FakeQuestionLoc,
                                                 move(LHS), 
                                                 FakeColonLoc,
                                                 move(RHS));
}
  
template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformImplicitCastExpr(ImplicitCastExpr *E) { 
  QualType T = getDerived().TransformType(E->getType());
  if (T.isNull())
    return SemaRef.ExprError();
  
  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      T == E->getType() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildImplicitCastExpr(T, E->getCastKind(),
                                              move(SubExpr), 
                                              E->isLvalueCast());
}
  
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformExplicitCastExpr(ExplicitCastExpr *E) { 
  assert(false && "Cannot transform abstract class");
  return SemaRef.Owned(E->Retain()); 
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCStyleCastExpr(CStyleCastExpr *E) { 
  QualType T;
  {
    // FIXME: Source location isn't quite accurate.
    SourceLocation TypeStartLoc 
      = SemaRef.PP.getLocForEndOfToken(E->getLParenLoc());
    TemporaryBase Rebase(*this, TypeStartLoc, DeclarationName());
    
    T = getDerived().TransformType(E->getTypeAsWritten());
    if (T.isNull())
      return SemaRef.ExprError();
  }
  
  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      T == E->getTypeAsWritten() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildCStyleCaseExpr(E->getLParenLoc(), T,
                                            E->getRParenLoc(),
                                            move(SubExpr));
}
  
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformCompoundLiteralExpr(CompoundLiteralExpr *E) {
  QualType T;
  {
    // FIXME: Source location isn't quite accurate.
    SourceLocation FakeTypeLoc 
      = SemaRef.PP.getLocForEndOfToken(E->getLParenLoc());
    TemporaryBase Rebase(*this, FakeTypeLoc, DeclarationName());
    
    T = getDerived().TransformType(E->getType());
    if (T.isNull())
      return SemaRef.ExprError();
  }
  
  OwningExprResult Init = getDerived().TransformExpr(E->getInitializer());
  if (Init.isInvalid())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      T == E->getType() &&
      Init.get() == E->getInitializer())
    return SemaRef.Owned(E->Retain()); 

  return getDerived().RebuildCompoundLiteralExpr(E->getLParenLoc(), T,
                                   /*FIXME:*/E->getInitializer()->getLocEnd(),
                                                 move(Init));
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformExtVectorElementExpr(ExtVectorElementExpr *E) { 
  OwningExprResult Base = getDerived().TransformExpr(E->getBase());
  if (Base.isInvalid())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      Base.get() == E->getBase())
    return SemaRef.Owned(E->Retain()); 
  
  // FIXME: Bad source location
  SourceLocation FakeOperatorLoc 
    = SemaRef.PP.getLocForEndOfToken(E->getBase()->getLocEnd());
  return getDerived().RebuildExtVectorElementExpr(move(Base), FakeOperatorLoc,
                                                  E->getAccessorLoc(),
                                                  E->getAccessor());
}
  
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformInitListExpr(InitListExpr *E) { 
  bool InitChanged = false;
  
  ASTOwningVector<&ActionBase::DeleteExpr, 4> Inits(SemaRef);
  for (unsigned I = 0, N = E->getNumInits(); I != N; ++I) {
    OwningExprResult Init = getDerived().TransformExpr(E->getInit(I));
    if (Init.isInvalid())
      return SemaRef.ExprError();
    
    InitChanged = InitChanged || Init.get() != E->getInit(I);
    Inits.push_back(Init.takeAs<Expr>());
  }
  
  if (!getDerived().AlwaysRebuild() && !InitChanged)
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildInitList(E->getLBraceLoc(), move_arg(Inits),
                                      E->getRBraceLoc());
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformDesignatedInitExpr(DesignatedInitExpr *E) { 
  Designation Desig;
  
  // transform the initializer value
  OwningExprResult Init = getDerived().TransformExpr(E->getInit());
  if (Init.isInvalid())
    return SemaRef.ExprError();
  
  // transform the designators.
  ASTOwningVector<&ActionBase::DeleteExpr, 4> ArrayExprs(SemaRef);
  bool ExprChanged = false;
  for (DesignatedInitExpr::designators_iterator D = E->designators_begin(),
                                             DEnd = E->designators_end();
       D != DEnd; ++D) {
    if (D->isFieldDesignator()) {
      Desig.AddDesignator(Designator::getField(D->getFieldName(),
                                               D->getDotLoc(),
                                               D->getFieldLoc()));
      continue;
    }
    
    if (D->isArrayDesignator()) {
      OwningExprResult Index = getDerived().TransformExpr(E->getArrayIndex(*D));
      if (Index.isInvalid())
        return SemaRef.ExprError();
      
      Desig.AddDesignator(Designator::getArray(Index.get(), 
                                               D->getLBracketLoc()));
      
      ExprChanged = ExprChanged || Init.get() != E->getArrayIndex(*D);
      ArrayExprs.push_back(Index.release());
      continue;
    }
    
    assert(D->isArrayRangeDesignator() && "New kind of designator?");
    OwningExprResult Start 
      = getDerived().TransformExpr(E->getArrayRangeStart(*D));
    if (Start.isInvalid())
      return SemaRef.ExprError();
    
    OwningExprResult End = getDerived().TransformExpr(E->getArrayRangeEnd(*D));
    if (End.isInvalid())
      return SemaRef.ExprError();
    
    Desig.AddDesignator(Designator::getArrayRange(Start.get(), 
                                                  End.get(),
                                                  D->getLBracketLoc(),
                                                  D->getEllipsisLoc()));
    
    ExprChanged = ExprChanged || Start.get() != E->getArrayRangeStart(*D) ||
      End.get() != E->getArrayRangeEnd(*D);
    
    ArrayExprs.push_back(Start.release());
    ArrayExprs.push_back(End.release());
  }
  
  if (!getDerived().AlwaysRebuild() &&
      Init.get() == E->getInit() &&
      !ExprChanged)
    return SemaRef.Owned(E->Retain());
  
  return getDerived().RebuildDesignatedInitExpr(Desig, move_arg(ArrayExprs),
                                                E->getEqualOrColonLoc(),
                                                E->usesGNUSyntax(), move(Init));
}
  
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformImplicitValueInitExpr(
                                                    ImplicitValueInitExpr *E) { 
  QualType T = getDerived().TransformType(E->getType());
  if (T.isNull())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      T == E->getType())
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildImplicitValueInitExpr(T);
}
  
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformVAArgExpr(VAArgExpr *E) {
  // FIXME: Do we want the type as written?
  QualType T;
  
  {
    // FIXME: Source location isn't quite accurate.
    TemporaryBase Rebase(*this, E->getBuiltinLoc(), DeclarationName());
    T = getDerived().TransformType(E->getType());
    if (T.isNull())
      return SemaRef.ExprError();
  }
  
  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      T == E->getType() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain());
  
  return getDerived().RebuildVAArgExpr(E->getBuiltinLoc(), move(SubExpr),
                                       T, E->getRParenLoc());
}

template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformParenListExpr(ParenListExpr *E) {
  bool ArgumentChanged = false;
  ASTOwningVector<&ActionBase::DeleteExpr, 4> Inits(SemaRef);
  for (unsigned I = 0, N = E->getNumExprs(); I != N; ++I) {
    OwningExprResult Init = getDerived().TransformExpr(E->getExpr(I));
    if (Init.isInvalid())
      return SemaRef.ExprError();
    
    ArgumentChanged = ArgumentChanged || Init.get() != E->getExpr(I);
    Inits.push_back(Init.takeAs<Expr>());
  }
  
  return getDerived().RebuildParenListExpr(E->getLParenLoc(),
                                           move_arg(Inits),
                                           E->getRParenLoc());
}
  
/// \brief Transform an address-of-label expression.
///
/// By default, the transformation of an address-of-label expression always
/// rebuilds the expression, so that the label identifier can be resolved to
/// the corresponding label statement by semantic analysis.
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformAddrLabelExpr(AddrLabelExpr *E) { 
  return getDerived().RebuildAddrLabelExpr(E->getAmpAmpLoc(), E->getLabelLoc(),
                                           E->getLabel());
}
  
template<typename Derived> 
Sema::OwningExprResult TreeTransform<Derived>::TransformStmtExpr(StmtExpr *E) {
  OwningStmtResult SubStmt 
    = getDerived().TransformCompoundStmt(E->getSubStmt(), true);
  if (SubStmt.isInvalid())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      SubStmt.get() == E->getSubStmt())
    return SemaRef.Owned(E->Retain());
  
  return getDerived().RebuildStmtExpr(E->getLParenLoc(), 
                                      move(SubStmt),
                                      E->getRParenLoc());
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformTypesCompatibleExpr(TypesCompatibleExpr *E) { 
  QualType T1, T2;
  {
    // FIXME: Source location isn't quite accurate.
    TemporaryBase Rebase(*this, E->getBuiltinLoc(), DeclarationName());
    
    T1 = getDerived().TransformType(E->getArgType1());
    if (T1.isNull())
      return SemaRef.ExprError();
    
    T2 = getDerived().TransformType(E->getArgType2());
    if (T2.isNull())
      return SemaRef.ExprError();
  }

  if (!getDerived().AlwaysRebuild() &&
      T1 == E->getArgType1() &&
      T2 == E->getArgType2())
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildTypesCompatibleExpr(E->getBuiltinLoc(),
                                                 T1, T2, E->getRParenLoc());
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformChooseExpr(ChooseExpr *E) { 
  OwningExprResult Cond = getDerived().TransformExpr(E->getCond());
  if (Cond.isInvalid())
    return SemaRef.ExprError();
  
  OwningExprResult LHS = getDerived().TransformExpr(E->getLHS());
  if (LHS.isInvalid())
    return SemaRef.ExprError();
  
  OwningExprResult RHS = getDerived().TransformExpr(E->getRHS());
  if (RHS.isInvalid())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      Cond.get() == E->getCond() &&
      LHS.get() == E->getLHS() &&
      RHS.get() == E->getRHS())
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildChooseExpr(E->getBuiltinLoc(),
                                        move(Cond), move(LHS), move(RHS),
                                        E->getRParenLoc());
}
  
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformGNUNullExpr(GNUNullExpr *E) { 
  return SemaRef.Owned(E->Retain()); 
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
  OwningExprResult Callee = getDerived().TransformExpr(E->getCallee());
  if (Callee.isInvalid())
    return SemaRef.ExprError();
  
  OwningExprResult First = getDerived().TransformExpr(E->getArg(0));
  if (First.isInvalid())
    return SemaRef.ExprError();

  OwningExprResult Second(SemaRef);
  if (E->getNumArgs() == 2) {
    Second = getDerived().TransformExpr(E->getArg(1));
    if (Second.isInvalid())
      return SemaRef.ExprError();
  }
  
  if (!getDerived().AlwaysRebuild() &&
      Callee.get() == E->getCallee() &&
      First.get() == E->getArg(0) &&
      (E->getNumArgs() != 2 || Second.get() == E->getArg(1)))    
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildCXXOperatorCallExpr(E->getOperator(),
                                                 E->getOperatorLoc(),
                                                 move(Callee), 
                                                 move(First),
                                                 move(Second));
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXMemberCallExpr(CXXMemberCallExpr *E) { 
  return getDerived().TransformCallExpr(E);
}
  
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformCXXNamedCastExpr(CXXNamedCastExpr *E) {
  QualType ExplicitTy;
  {
    // FIXME: Source location isn't quite accurate.
    SourceLocation TypeStartLoc 
      = SemaRef.PP.getLocForEndOfToken(E->getOperatorLoc());
    TemporaryBase Rebase(*this, TypeStartLoc, DeclarationName());
  
    ExplicitTy = getDerived().TransformType(E->getTypeAsWritten());
    if (ExplicitTy.isNull())
      return SemaRef.ExprError();
  }
  
  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      ExplicitTy == E->getTypeAsWritten() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain()); 
  
  // FIXME: Poor source location information here.
  SourceLocation FakeLAngleLoc 
    = SemaRef.PP.getLocForEndOfToken(E->getOperatorLoc());
  SourceLocation FakeRAngleLoc = E->getSubExpr()->getSourceRange().getBegin();
  SourceLocation FakeRParenLoc
    = SemaRef.PP.getLocForEndOfToken(
                                  E->getSubExpr()->getSourceRange().getEnd());
  return getDerived().RebuildCXXNamedCastExpr(E->getOperatorLoc(),
                                              E->getStmtClass(), 
                                              FakeLAngleLoc,
                                              ExplicitTy,
                                              FakeRAngleLoc,
                                              FakeRAngleLoc,
                                              move(SubExpr),
                                              FakeRParenLoc);
}
  
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformCXXStaticCastExpr(CXXStaticCastExpr *E) { 
  return getDerived().TransformCXXNamedCastExpr(E);
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXDynamicCastExpr(CXXDynamicCastExpr *E) { 
  return getDerived().TransformCXXNamedCastExpr(E);
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXReinterpretCastExpr(
                                                CXXReinterpretCastExpr *E) { 
  return getDerived().TransformCXXNamedCastExpr(E);
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXConstCastExpr(CXXConstCastExpr *E) { 
  return getDerived().TransformCXXNamedCastExpr(E);
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXFunctionalCastExpr(
                                                   CXXFunctionalCastExpr *E) { 
  QualType ExplicitTy;
  {
    TemporaryBase Rebase(*this, E->getTypeBeginLoc(), DeclarationName());
    
    ExplicitTy = getDerived().TransformType(E->getTypeAsWritten());
    if (ExplicitTy.isNull())
      return SemaRef.ExprError();
  }
  
  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      ExplicitTy == E->getTypeAsWritten() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain()); 
  
  // FIXME: The end of the type's source range is wrong
  return getDerived().RebuildCXXFunctionalCastExpr(
                                  /*FIXME:*/SourceRange(E->getTypeBeginLoc()),
                                                   ExplicitTy,
                                      /*FIXME:*/E->getSubExpr()->getLocStart(),
                                                   move(SubExpr),
                                                   E->getRParenLoc());
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXTypeidExpr(CXXTypeidExpr *E) { 
  if (E->isTypeOperand()) {
    TemporaryBase Rebase(*this, /*FIXME*/E->getLocStart(), DeclarationName());
    
    QualType T = getDerived().TransformType(E->getTypeOperand());
    if (T.isNull())
      return SemaRef.ExprError();
    
    if (!getDerived().AlwaysRebuild() &&
        T == E->getTypeOperand())
      return SemaRef.Owned(E->Retain());
    
    return getDerived().RebuildCXXTypeidExpr(E->getLocStart(),
                                             /*FIXME:*/E->getLocStart(),
                                             T,
                                             E->getLocEnd());
  }
  
  // We don't know whether the expression is potentially evaluated until
  // after we perform semantic analysis, so the expression is potentially
  // potentially evaluated.
  EnterExpressionEvaluationContext Unevaluated(SemaRef, 
                                      Action::PotentiallyPotentiallyEvaluated);
  
  OwningExprResult SubExpr = getDerived().TransformExpr(E->getExprOperand());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      SubExpr.get() == E->getExprOperand())
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildCXXTypeidExpr(E->getLocStart(),
                                           /*FIXME:*/E->getLocStart(),
                                           move(SubExpr),
                                           E->getLocEnd());
}

template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::TransformCXXBoolLiteralExpr(CXXBoolLiteralExpr *E) { 
  return SemaRef.Owned(E->Retain()); 
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXNullPtrLiteralExpr(
                                                  CXXNullPtrLiteralExpr *E) { 
  return SemaRef.Owned(E->Retain()); 
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXThisExpr(CXXThisExpr *E) {
  TemporaryBase Rebase(*this, E->getLocStart(), DeclarationName());
    
  QualType T = getDerived().TransformType(E->getType());
  if (T.isNull())
    return SemaRef.ExprError();
    
  if (!getDerived().AlwaysRebuild() &&
      T == E->getType())
    return SemaRef.Owned(E->Retain());
 
  return getDerived().RebuildCXXThisExpr(E->getLocStart(), T);
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXThrowExpr(CXXThrowExpr *E) { 
  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain()); 

  return getDerived().RebuildCXXThrowExpr(E->getThrowLoc(), move(SubExpr));
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXDefaultArgExpr(CXXDefaultArgExpr *E) { 
  ParmVarDecl *Param 
    = cast_or_null<ParmVarDecl>(getDerived().TransformDecl(E->getParam()));
  if (!Param)
    return SemaRef.ExprError();
  
  if (getDerived().AlwaysRebuild() &&
      Param == E->getParam())
    return SemaRef.Owned(E->Retain());
  
  return getDerived().RebuildCXXDefaultArgExpr(Param);
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXZeroInitValueExpr(CXXZeroInitValueExpr *E) {
  TemporaryBase Rebase(*this, E->getTypeBeginLoc(), DeclarationName());

  QualType T = getDerived().TransformType(E->getType());
  if (T.isNull())
    return SemaRef.ExprError();
    
  if (!getDerived().AlwaysRebuild() &&
      T == E->getType())
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildCXXZeroInitValueExpr(E->getTypeBeginLoc(), 
                                                /*FIXME:*/E->getTypeBeginLoc(),
                                                  T,
                                                  E->getRParenLoc());
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXConditionDeclExpr(CXXConditionDeclExpr *E) {
  VarDecl *Var 
    = cast_or_null<VarDecl>(getDerived().TransformDefinition(E->getVarDecl()));
  if (!Var)
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      Var == E->getVarDecl())    
    return SemaRef.Owned(E->Retain());
  
  return getDerived().RebuildCXXConditionDeclExpr(E->getStartLoc(), 
                                                  /*FIXME:*/E->getStartLoc(),
                                                  Var);
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXNewExpr(CXXNewExpr *E) { 
  // Transform the type that we're allocating
  TemporaryBase Rebase(*this, E->getLocStart(), DeclarationName());
  QualType AllocType = getDerived().TransformType(E->getAllocatedType());
  if (AllocType.isNull())
    return SemaRef.ExprError();
  
  // Transform the size of the array we're allocating (if any).
  OwningExprResult ArraySize = getDerived().TransformExpr(E->getArraySize());
  if (ArraySize.isInvalid())
    return SemaRef.ExprError();
  
  // Transform the placement arguments (if any).
  bool ArgumentChanged = false;
  ASTOwningVector<&ActionBase::DeleteExpr> PlacementArgs(SemaRef);
  for (unsigned I = 0, N = E->getNumPlacementArgs(); I != N; ++I) {
    OwningExprResult Arg = getDerived().TransformExpr(E->getPlacementArg(I));
    if (Arg.isInvalid())
      return SemaRef.ExprError();
    
    ArgumentChanged = ArgumentChanged || Arg.get() != E->getPlacementArg(I);
    PlacementArgs.push_back(Arg.take());
  }
  
  // transform the constructor arguments (if any).
  ASTOwningVector<&ActionBase::DeleteExpr> ConstructorArgs(SemaRef);
  for (unsigned I = 0, N = E->getNumConstructorArgs(); I != N; ++I) {
    OwningExprResult Arg = getDerived().TransformExpr(E->getConstructorArg(I));
    if (Arg.isInvalid())
      return SemaRef.ExprError();
    
    ArgumentChanged = ArgumentChanged || Arg.get() != E->getConstructorArg(I);
    ConstructorArgs.push_back(Arg.take());
  }
  
  if (!getDerived().AlwaysRebuild() &&
      AllocType == E->getAllocatedType() &&
      ArraySize.get() == E->getArraySize() &&
      !ArgumentChanged)
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildCXXNewExpr(E->getLocStart(),
                                        E->isGlobalNew(),
                                        /*FIXME:*/E->getLocStart(),
                                        move_arg(PlacementArgs),
                                        /*FIXME:*/E->getLocStart(),
                                        E->isParenTypeId(),
                                        AllocType,
                                        /*FIXME:*/E->getLocStart(),
                                        /*FIXME:*/SourceRange(),
                                        move(ArraySize),
                                        /*FIXME:*/E->getLocStart(),
                                        move_arg(ConstructorArgs),
                                        E->getLocEnd());  
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXDeleteExpr(CXXDeleteExpr *E) { 
  OwningExprResult Operand = getDerived().TransformExpr(E->getArgument());
  if (Operand.isInvalid())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      Operand.get() == E->getArgument())    
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildCXXDeleteExpr(E->getLocStart(),
                                           E->isGlobalDelete(),
                                           E->isArrayForm(),
                                           move(Operand));
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformUnresolvedFunctionNameExpr(
                                              UnresolvedFunctionNameExpr *E) { 
  // There is no transformation we can apply to an unresolved function name.
  return SemaRef.Owned(E->Retain()); 
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformUnaryTypeTraitExpr(UnaryTypeTraitExpr *E) { 
  TemporaryBase Rebase(*this, /*FIXME*/E->getLocStart(), DeclarationName());
  
  QualType T = getDerived().TransformType(E->getQueriedType());
  if (T.isNull())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      T == E->getQueriedType())
    return SemaRef.Owned(E->Retain());
    
  // FIXME: Bad location information
  SourceLocation FakeLParenLoc
    = SemaRef.PP.getLocForEndOfToken(E->getLocStart());
  
  return getDerived().RebuildUnaryTypeTrait(E->getTrait(), 
                                            E->getLocStart(),
                                            /*FIXME:*/FakeLParenLoc,
                                            T,
                                            E->getLocEnd());
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformQualifiedDeclRefExpr(QualifiedDeclRefExpr *E) {
  NestedNameSpecifier *NNS
    = getDerived().TransformNestedNameSpecifier(E->getQualifier(),
                                                E->getQualifierRange());
  if (!NNS)
    return SemaRef.ExprError();
  
  NamedDecl *ND 
    = dyn_cast_or_null<NamedDecl>(getDerived().TransformDecl(E->getDecl()));
  if (!ND)
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() && 
      NNS == E->getQualifier() &&
      ND == E->getDecl())
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildQualifiedDeclRefExpr(NNS, 
                                                  E->getQualifierRange(),
                                                  ND,
                                                  E->getLocation(),
                                                  /*FIXME:*/false);
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformUnresolvedDeclRefExpr(
                                                    UnresolvedDeclRefExpr *E) { 
  NestedNameSpecifier *NNS
  = getDerived().TransformNestedNameSpecifier(E->getQualifier(),
                                              E->getQualifierRange());
  if (!NNS)
    return SemaRef.ExprError();
  
  // FIXME: Transform the declaration name
  DeclarationName Name = E->getDeclName();
  
  if (!getDerived().AlwaysRebuild() && 
      NNS == E->getQualifier() &&
      Name == E->getDeclName())
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildUnresolvedDeclRefExpr(NNS, 
                                                   E->getQualifierRange(),
                                                   Name,
                                                   E->getLocation(),
                                                   /*FIXME:*/false);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformTemplateIdRefExpr(TemplateIdRefExpr *E) { 
  TemplateName Template 
    = getDerived().TransformTemplateName(E->getTemplateName());
  if (Template.isNull())
    return SemaRef.ExprError();
  
  llvm::SmallVector<TemplateArgument, 4> TransArgs;
  for (unsigned I = 0, N = E->getNumTemplateArgs(); I != N; ++I) {
    TemplateArgument TransArg 
      = getDerived().TransformTemplateArgument(E->getTemplateArgs()[I]);
    if (TransArg.isNull())
      return SemaRef.ExprError();

    TransArgs.push_back(TransArg);
  }

  // FIXME: Would like to avoid rebuilding if nothing changed, but we can't
  // compare template arguments (yet).
  
  // FIXME: It's possible that we'll find out now that the template name 
  // actually refers to a type, in which case the caller is actually dealing
  // with a functional cast. Give a reasonable error message!
  return getDerived().RebuildTemplateIdExpr(Template, E->getTemplateNameLoc(),
                                            E->getLAngleLoc(),
                                            TransArgs.data(),
                                            TransArgs.size(),
                                            E->getRAngleLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXConstructExpr(CXXConstructExpr *E) { 
  TemporaryBase Rebase(*this, /*FIXME*/E->getLocStart(), DeclarationName());

  QualType T = getDerived().TransformType(E->getType());
  if (T.isNull())
    return SemaRef.ExprError();

  CXXConstructorDecl *Constructor
    = cast_or_null<CXXConstructorDecl>(
                              getDerived().TransformDecl(E->getConstructor()));
  if (!Constructor)
    return SemaRef.ExprError();
  
  bool ArgumentChanged = false;
  ASTOwningVector<&ActionBase::DeleteExpr> Args(SemaRef);
  for (CXXConstructExpr::arg_iterator Arg = E->arg_begin(), 
       ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg) {
    OwningExprResult TransArg = getDerived().TransformExpr(*Arg);
    if (TransArg.isInvalid())
      return SemaRef.ExprError();
    
    ArgumentChanged = ArgumentChanged || TransArg.get() != *Arg;
    Args.push_back(TransArg.takeAs<Expr>());
  }

  if (!getDerived().AlwaysRebuild() &&
      T == E->getType() &&
      Constructor == E->getConstructor() &&
      !ArgumentChanged)
    return SemaRef.Owned(E->Retain());
  
  return getDerived().RebuildCXXConstructExpr(T, Constructor, E->isElidable(),
                                              move_arg(Args));
}
  
/// \brief Transform a C++ temporary-binding expression.
///
/// The transformation of a temporary-binding expression always attempts to 
/// bind a new temporary variable to its subexpression, even if the 
/// subexpression itself did not change, because the temporary variable itself
/// must be unique.
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();
  
  return SemaRef.MaybeBindToTemporary(SubExpr.takeAs<Expr>());
}
  
/// \brief Transform a C++ expression that contains temporaries that should 
/// be destroyed after the expression is evaluated.
///
/// The transformation of a full expression always attempts to build a new 
/// CXXExprWithTemporaries expression, even if the 
/// subexpression itself did not change, because it will need to capture the
/// the new temporary variables introduced in the subexpression.
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXExprWithTemporaries(
                                                CXXExprWithTemporaries *E) { 
  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();
  
  return SemaRef.Owned(
           SemaRef.MaybeCreateCXXExprWithTemporaries(SubExpr.takeAs<Expr>(),
                                               E->shouldDestroyTemporaries()));
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXTemporaryObjectExpr(
                                                  CXXTemporaryObjectExpr *E) { 
  TemporaryBase Rebase(*this, E->getTypeBeginLoc(), DeclarationName());
  QualType T = getDerived().TransformType(E->getType());
  if (T.isNull())
    return SemaRef.ExprError();
  
  CXXConstructorDecl *Constructor
    = cast_or_null<CXXConstructorDecl>(
                            getDerived().TransformDecl(E->getConstructor()));
  if (!Constructor)
    return SemaRef.ExprError();
  
  bool ArgumentChanged = false;
  ASTOwningVector<&ActionBase::DeleteExpr> Args(SemaRef);
  Args.reserve(E->getNumArgs());
  for (CXXTemporaryObjectExpr::arg_iterator Arg = E->arg_begin(), 
                                         ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg) {
    OwningExprResult TransArg = getDerived().TransformExpr(*Arg);
    if (TransArg.isInvalid())
      return SemaRef.ExprError();
    
    ArgumentChanged = ArgumentChanged || TransArg.get() != *Arg;
    Args.push_back((Expr *)TransArg.release());
  }
  
  if (!getDerived().AlwaysRebuild() &&
      T == E->getType() &&
      Constructor == E->getConstructor() &&
      !ArgumentChanged)
    return SemaRef.Owned(E->Retain());
  
  // FIXME: Bogus location information
  SourceLocation CommaLoc;
  if (Args.size() > 1) {
    Expr *First = (Expr *)Args[0];
    CommaLoc 
      = SemaRef.PP.getLocForEndOfToken(First->getSourceRange().getEnd());
  }
  return getDerived().RebuildCXXTemporaryObjectExpr(E->getTypeBeginLoc(),
                                                    T,
                                                /*FIXME:*/E->getTypeBeginLoc(),
                                                    move_arg(Args),
                                                    &CommaLoc,
                                                    E->getLocEnd());
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXUnresolvedConstructExpr(
                                            CXXUnresolvedConstructExpr *E) { 
  TemporaryBase Rebase(*this, E->getTypeBeginLoc(), DeclarationName());
  QualType T = getDerived().TransformType(E->getTypeAsWritten());
  if (T.isNull())
    return SemaRef.ExprError();
  
  bool ArgumentChanged = false;
  ASTOwningVector<&ActionBase::DeleteExpr> Args(SemaRef);
  llvm::SmallVector<SourceLocation, 8> FakeCommaLocs;
  for (CXXUnresolvedConstructExpr::arg_iterator Arg = E->arg_begin(),
                                             ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg) {
    OwningExprResult TransArg = getDerived().TransformExpr(*Arg);
    if (TransArg.isInvalid())
      return SemaRef.ExprError();
    
    ArgumentChanged = ArgumentChanged || TransArg.get() != *Arg;
    FakeCommaLocs.push_back(
                        SemaRef.PP.getLocForEndOfToken((*Arg)->getLocEnd()));
    Args.push_back(TransArg.takeAs<Expr>());
  }
  
  if (!getDerived().AlwaysRebuild() &&
      T == E->getTypeAsWritten() &&
      !ArgumentChanged)
    return SemaRef.Owned(E->Retain()); 
  
  // FIXME: we're faking the locations of the commas
  return getDerived().RebuildCXXUnresolvedConstructExpr(E->getTypeBeginLoc(),
                                                        T,
                                                        E->getLParenLoc(),
                                                        move_arg(Args),
                                                        FakeCommaLocs.data(),
                                                        E->getRParenLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXUnresolvedMemberExpr(
                                                  CXXUnresolvedMemberExpr *E) { 
  // Transform the base of the expression.
  OwningExprResult Base = getDerived().TransformExpr(E->getBase());
  if (Base.isInvalid())
    return SemaRef.ExprError();
  
  // FIXME: Transform the declaration name
  DeclarationName Name = E->getMember();
  
  if (!getDerived().AlwaysRebuild() &&
      Base.get() == E->getBase() &&
      Name == E->getMember())
    return SemaRef.Owned(E->Retain()); 
      
  return getDerived().RebuildCXXUnresolvedMemberExpr(move(Base),
                                                     E->isArrow(),
                                                     E->getOperatorLoc(),
                                                     E->getMember(),
                                                     E->getMemberLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCStringLiteral(ObjCStringLiteral *E) { 
  return SemaRef.Owned(E->Retain()); 
}

template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformObjCEncodeExpr(ObjCEncodeExpr *E) { 
  // FIXME: poor source location
  TemporaryBase Rebase(*this, E->getAtLoc(), DeclarationName());
  QualType EncodedType = getDerived().TransformType(E->getEncodedType());
  if (EncodedType.isNull())
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      EncodedType == E->getEncodedType())
    return SemaRef.Owned(E->Retain()); 

  return getDerived().RebuildObjCEncodeExpr(E->getAtLoc(),
                                            EncodedType,
                                            E->getRParenLoc());
}
  
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCMessageExpr(ObjCMessageExpr *E) { 
  // FIXME: Implement this!
  assert(false && "Cannot transform Objective-C expressions yet");
  return SemaRef.Owned(E->Retain()); 
}

template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformObjCSelectorExpr(ObjCSelectorExpr *E) { 
  return SemaRef.Owned(E->Retain()); 
}

template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformObjCProtocolExpr(ObjCProtocolExpr *E) { 
  ObjCProtocolDecl *Protocol 
    = cast_or_null<ObjCProtocolDecl>(
                                getDerived().TransformDecl(E->getProtocol()));
  if (!Protocol)
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      Protocol == E->getProtocol())
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildObjCProtocolExpr(Protocol,
                                              E->getAtLoc(),
                                              /*FIXME:*/E->getAtLoc(),
                                              /*FIXME:*/E->getAtLoc(),
                                              E->getRParenLoc());
                                              
}

template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformObjCIvarRefExpr(ObjCIvarRefExpr *E) { 
  // FIXME: Implement this!
  assert(false && "Cannot transform Objective-C expressions yet");
  return SemaRef.Owned(E->Retain()); 
}

template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
  // FIXME: Implement this!
  assert(false && "Cannot transform Objective-C expressions yet");
  return SemaRef.Owned(E->Retain()); 
}

template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformObjCImplicitSetterGetterRefExpr(
                                          ObjCImplicitSetterGetterRefExpr *E) { 
  // FIXME: Implement this!
  assert(false && "Cannot transform Objective-C expressions yet");
  return SemaRef.Owned(E->Retain()); 
}

template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformObjCSuperExpr(ObjCSuperExpr *E) { 
  // FIXME: Implement this!
  assert(false && "Cannot transform Objective-C expressions yet");
  return SemaRef.Owned(E->Retain()); 
}

template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformObjCIsaExpr(ObjCIsaExpr *E) { 
  // FIXME: Implement this!
  assert(false && "Cannot transform Objective-C expressions yet");
  return SemaRef.Owned(E->Retain()); 
}

template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformShuffleVectorExpr(ShuffleVectorExpr *E) { 
  bool ArgumentChanged = false;
  ASTOwningVector<&ActionBase::DeleteExpr> SubExprs(SemaRef);
  for (unsigned I = 0, N = E->getNumSubExprs(); I != N; ++I) {
    OwningExprResult SubExpr = getDerived().TransformExpr(E->getExpr(I));
    if (SubExpr.isInvalid())
      return SemaRef.ExprError();
    
    ArgumentChanged = ArgumentChanged || SubExpr.get() != E->getExpr(I);
    SubExprs.push_back(SubExpr.takeAs<Expr>());
  }
  
  if (!getDerived().AlwaysRebuild() &&
      !ArgumentChanged)
    return SemaRef.Owned(E->Retain()); 
  
  return getDerived().RebuildShuffleVectorExpr(E->getBuiltinLoc(),
                                               move_arg(SubExprs),
                                               E->getRParenLoc());
}

template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformBlockExpr(BlockExpr *E) { 
  // FIXME: Implement this!
  assert(false && "Cannot transform block expressions yet");
  return SemaRef.Owned(E->Retain()); 
}

template<typename Derived> 
Sema::OwningExprResult 
TreeTransform<Derived>::TransformBlockDeclRefExpr(BlockDeclRefExpr *E) { 
  // FIXME: Implement this!
  assert(false && "Cannot transform block-related expressions yet");
  return SemaRef.Owned(E->Retain()); 
}
  
//===----------------------------------------------------------------------===//
// Type reconstruction
//===----------------------------------------------------------------------===//

template<typename Derived> 
QualType TreeTransform<Derived>::RebuildPointerType(QualType PointeeType) {
  return SemaRef.BuildPointerType(PointeeType, 0, 
                                  getDerived().getBaseLocation(),
                                  getDerived().getBaseEntity());
}

template<typename Derived> 
QualType TreeTransform<Derived>::RebuildBlockPointerType(QualType PointeeType) {
  return SemaRef.BuildBlockPointerType(PointeeType, 0, 
                                       getDerived().getBaseLocation(),
                                       getDerived().getBaseEntity());
}

template<typename Derived> 
QualType 
TreeTransform<Derived>::RebuildLValueReferenceType(QualType ReferentType) {
  return SemaRef.BuildReferenceType(ReferentType, true, 0, 
                                    getDerived().getBaseLocation(),
                                    getDerived().getBaseEntity());
}

template<typename Derived> 
QualType 
TreeTransform<Derived>::RebuildRValueReferenceType(QualType ReferentType) {
  return SemaRef.BuildReferenceType(ReferentType, false, 0, 
                                    getDerived().getBaseLocation(),
                                    getDerived().getBaseEntity());
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildMemberPointerType(QualType PointeeType, 
                                                          QualType ClassType) {
  return SemaRef.BuildMemberPointerType(PointeeType, ClassType, 0, 
                                        getDerived().getBaseLocation(),
                                        getDerived().getBaseEntity());
}

template<typename Derived>
QualType 
TreeTransform<Derived>::RebuildArrayType(QualType ElementType,
                                         ArrayType::ArraySizeModifier SizeMod,
                                         const llvm::APInt *Size,
                                         Expr *SizeExpr,
                                         unsigned IndexTypeQuals,
                                         SourceRange BracketsRange) {
  if (SizeExpr || !Size)
    return SemaRef.BuildArrayType(ElementType, SizeMod, SizeExpr,
                                  IndexTypeQuals, BracketsRange,
                                  getDerived().getBaseEntity());
  
  QualType Types[] = { 
    SemaRef.Context.UnsignedCharTy, SemaRef.Context.UnsignedShortTy, 
    SemaRef.Context.UnsignedIntTy, SemaRef.Context.UnsignedLongTy, 
    SemaRef.Context.UnsignedLongLongTy, SemaRef.Context.UnsignedInt128Ty 
  };
  const unsigned NumTypes = sizeof(Types) / sizeof(QualType);
  QualType SizeType;
  for (unsigned I = 0; I != NumTypes; ++I)
    if (Size->getBitWidth() == SemaRef.Context.getIntWidth(Types[I])) {
      SizeType = Types[I];
      break;
    }
  
  if (SizeType.isNull())
    SizeType = SemaRef.Context.getFixedWidthIntType(Size->getBitWidth(), false);
  
  IntegerLiteral ArraySize(*Size, SizeType, /*FIXME*/BracketsRange.getBegin());
  return SemaRef.BuildArrayType(ElementType, SizeMod, &ArraySize, 
                                IndexTypeQuals, BracketsRange,
                                getDerived().getBaseEntity());    
}
  
template<typename Derived>
QualType 
TreeTransform<Derived>::RebuildConstantArrayType(QualType ElementType, 
                                                 ArrayType::ArraySizeModifier SizeMod,
                                                 const llvm::APInt &Size,
                                                 unsigned IndexTypeQuals) {
  return getDerived().RebuildArrayType(ElementType, SizeMod, &Size, 0, 
                                        IndexTypeQuals, SourceRange());
}

template<typename Derived>
QualType 
TreeTransform<Derived>::RebuildConstantArrayWithExprType(QualType ElementType, 
                                          ArrayType::ArraySizeModifier SizeMod,
                                                      const llvm::APInt &Size,
                                                         Expr *SizeExpr,
                                                      unsigned IndexTypeQuals,
                                                    SourceRange BracketsRange) {
  return getDerived().RebuildArrayType(ElementType, SizeMod, &Size, SizeExpr, 
                                       IndexTypeQuals, BracketsRange);
}

template<typename Derived>
QualType 
TreeTransform<Derived>::RebuildConstantArrayWithoutExprType(
                                                        QualType ElementType, 
                                          ArrayType::ArraySizeModifier SizeMod,
                                                       const llvm::APInt &Size,
                                                     unsigned IndexTypeQuals) {
  return getDerived().RebuildArrayType(ElementType, SizeMod, &Size, 0, 
                                       IndexTypeQuals, SourceRange());
}

template<typename Derived>
QualType 
TreeTransform<Derived>::RebuildIncompleteArrayType(QualType ElementType, 
                                          ArrayType::ArraySizeModifier SizeMod,
                                                 unsigned IndexTypeQuals) {
  return getDerived().RebuildArrayType(ElementType, SizeMod, 0, 0, 
                                       IndexTypeQuals, SourceRange());
}
  
template<typename Derived>
QualType 
TreeTransform<Derived>::RebuildVariableArrayType(QualType ElementType, 
                                          ArrayType::ArraySizeModifier SizeMod,
                                                 ExprArg SizeExpr,
                                                 unsigned IndexTypeQuals,
                                                 SourceRange BracketsRange) {
  return getDerived().RebuildArrayType(ElementType, SizeMod, 0, 
                                       SizeExpr.takeAs<Expr>(),
                                       IndexTypeQuals, BracketsRange);
}

template<typename Derived>
QualType 
TreeTransform<Derived>::RebuildDependentSizedArrayType(QualType ElementType, 
                                          ArrayType::ArraySizeModifier SizeMod,
                                                       ExprArg SizeExpr,
                                                       unsigned IndexTypeQuals,
                                                   SourceRange BracketsRange) {
  return getDerived().RebuildArrayType(ElementType, SizeMod, 0, 
                                       SizeExpr.takeAs<Expr>(),
                                       IndexTypeQuals, BracketsRange);
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildVectorType(QualType ElementType,
                                                   unsigned NumElements) {
  // FIXME: semantic checking!
  return SemaRef.Context.getVectorType(ElementType, NumElements);
}
  
template<typename Derived>
QualType TreeTransform<Derived>::RebuildExtVectorType(QualType ElementType,
                                                      unsigned NumElements,
                                                 SourceLocation AttributeLoc) {
  llvm::APInt numElements(SemaRef.Context.getIntWidth(SemaRef.Context.IntTy),
                          NumElements, true);
  IntegerLiteral *VectorSize
    = new (SemaRef.Context) IntegerLiteral(numElements, SemaRef.Context.IntTy, 
                                           AttributeLoc);
  return SemaRef.BuildExtVectorType(ElementType, SemaRef.Owned(VectorSize),
                                    AttributeLoc);
}
  
template<typename Derived>
QualType 
TreeTransform<Derived>::RebuildDependentSizedExtVectorType(QualType ElementType, 
                                                           ExprArg SizeExpr,
                                                  SourceLocation AttributeLoc) {
  return SemaRef.BuildExtVectorType(ElementType, move(SizeExpr), AttributeLoc);
}
 
template<typename Derived>
QualType TreeTransform<Derived>::RebuildFunctionProtoType(QualType T,
                                                          QualType *ParamTypes, 
                                                        unsigned NumParamTypes,
                                                          bool Variadic, 
                                                          unsigned Quals) {
  return SemaRef.BuildFunctionType(T, ParamTypes, NumParamTypes, Variadic, 
                                   Quals,
                                   getDerived().getBaseLocation(),
                                   getDerived().getBaseEntity());
}
  
template<typename Derived>
QualType TreeTransform<Derived>::RebuildTypeOfExprType(ExprArg E) {
  return SemaRef.BuildTypeofExprType(E.takeAs<Expr>());
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildTypeOfType(QualType Underlying) {
  return SemaRef.Context.getTypeOfType(Underlying);
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildDecltypeType(ExprArg E) {
  return SemaRef.BuildDecltypeType(E.takeAs<Expr>());
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildTemplateSpecializationType(
                                                        TemplateName Template,
                                                 const TemplateArgument *Args,
                                                           unsigned NumArgs) {
  // FIXME: Missing source locations for the template name, <, >.
  return SemaRef.CheckTemplateIdType(Template, getDerived().getBaseLocation(),
                                     SourceLocation(), Args, NumArgs, 
                                     SourceLocation());  
}
  
template<typename Derived>
NestedNameSpecifier *
TreeTransform<Derived>::RebuildNestedNameSpecifier(NestedNameSpecifier *Prefix,
                                                   SourceRange Range,
                                                   IdentifierInfo &II) {
  CXXScopeSpec SS;
  // FIXME: The source location information is all wrong.
  SS.setRange(Range);
  SS.setScopeRep(Prefix);
  return static_cast<NestedNameSpecifier *>(
                    SemaRef.ActOnCXXNestedNameSpecifier(0, SS, Range.getEnd(), 
                                                        Range.getEnd(), II));
}

template<typename Derived>
NestedNameSpecifier *
TreeTransform<Derived>::RebuildNestedNameSpecifier(NestedNameSpecifier *Prefix,
                                                   SourceRange Range,
                                                   NamespaceDecl *NS) {
  return NestedNameSpecifier::Create(SemaRef.Context, Prefix, NS);
}

template<typename Derived>
NestedNameSpecifier *
TreeTransform<Derived>::RebuildNestedNameSpecifier(NestedNameSpecifier *Prefix,
                                                   SourceRange Range,
                                                   bool TemplateKW,
                                                   QualType T) {
  if (T->isDependentType() || T->isRecordType() ||
      (SemaRef.getLangOptions().CPlusPlus0x && T->isEnumeralType())) {
    assert(T.getCVRQualifiers() == 0 && "Can't get cv-qualifiers here");
    return NestedNameSpecifier::Create(SemaRef.Context, Prefix, TemplateKW,
                                       T.getTypePtr());
  }
  
  SemaRef.Diag(Range.getBegin(), diag::err_nested_name_spec_non_tag) << T;
  return 0;
}
  
template<typename Derived>
TemplateName 
TreeTransform<Derived>::RebuildTemplateName(NestedNameSpecifier *Qualifier,
                                            bool TemplateKW,
                                            TemplateDecl *Template) {
  return SemaRef.Context.getQualifiedTemplateName(Qualifier, TemplateKW, 
                                                  Template);
}

template<typename Derived>
TemplateName 
TreeTransform<Derived>::RebuildTemplateName(NestedNameSpecifier *Qualifier,
                                            bool TemplateKW,
                                            OverloadedFunctionDecl *Ovl) {
  return SemaRef.Context.getQualifiedTemplateName(Qualifier, TemplateKW, Ovl);
}

template<typename Derived>
TemplateName 
TreeTransform<Derived>::RebuildTemplateName(NestedNameSpecifier *Qualifier,
                                            const IdentifierInfo &II) {
  if (Qualifier->isDependent())
    return SemaRef.Context.getDependentTemplateName(Qualifier, &II);
  
  // Somewhat redundant with ActOnDependentTemplateName.
  CXXScopeSpec SS;
  SS.setRange(SourceRange(getDerived().getBaseLocation()));
  SS.setScopeRep(Qualifier);
  Sema::TemplateTy Template;
  TemplateNameKind TNK = SemaRef.isTemplateName(II, 0, Template, &SS);
  if (TNK == TNK_Non_template) {
    SemaRef.Diag(getDerived().getBaseLocation(), 
                 diag::err_template_kw_refers_to_non_template)
      << &II;
    return TemplateName();
  } else if (TNK == TNK_Function_template) {
    SemaRef.Diag(getDerived().getBaseLocation(), 
                 diag::err_template_kw_refers_to_non_template)
      << &II;
    return TemplateName();
  }
  
  return Template.getAsVal<TemplateName>();  
}
 
template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::RebuildCXXOperatorCallExpr(OverloadedOperatorKind Op,
                                                   SourceLocation OpLoc,
                                                   ExprArg Callee,
                                                   ExprArg First,
                                                   ExprArg Second) {
  Expr *FirstExpr = (Expr *)First.get();
  Expr *SecondExpr = (Expr *)Second.get();
  bool isPostIncDec = SecondExpr && (Op == OO_PlusPlus || Op == OO_MinusMinus);
    
  // Determine whether this should be a builtin operation.
  if (SecondExpr == 0 || isPostIncDec) {
    if (!FirstExpr->getType()->isOverloadableType()) {
      // The argument is not of overloadable type, so try to create a
      // built-in unary operation.
      UnaryOperator::Opcode Opc 
        = UnaryOperator::getOverloadedOpcode(Op, isPostIncDec);
      
      return getSema().CreateBuiltinUnaryOp(OpLoc, Opc, move(First));
    }
  } else {
    if (!FirstExpr->getType()->isOverloadableType() && 
        !SecondExpr->getType()->isOverloadableType()) {
      // Neither of the arguments is an overloadable type, so try to
      // create a built-in binary operation.
      BinaryOperator::Opcode Opc = BinaryOperator::getOverloadedOpcode(Op);
      OwningExprResult Result 
        = SemaRef.CreateBuiltinBinOp(OpLoc, Opc, FirstExpr, SecondExpr);
      if (Result.isInvalid())
        return SemaRef.ExprError();
      
      First.release();
      Second.release();
      return move(Result);
    }
  }
  
  // Compute the transformed set of functions (and function templates) to be 
  // used during overload resolution.
  Sema::FunctionSet Functions;
  
  DeclRefExpr *DRE = cast<DeclRefExpr>((Expr *)Callee.get());
  OverloadedFunctionDecl *Overloads 
    = cast<OverloadedFunctionDecl>(DRE->getDecl());
  
  // FIXME: Do we have to check
  // IsAcceptableNonMemberOperatorCandidate for each of these?
  for (OverloadedFunctionDecl::function_iterator 
       F = Overloads->function_begin(),
       FEnd = Overloads->function_end();
       F != FEnd; ++F)
    Functions.insert(*F);
  
  // Add any functions found via argument-dependent lookup.
  Expr *Args[2] = { FirstExpr, SecondExpr };
  unsigned NumArgs = 1 + (SecondExpr != 0);
  DeclarationName OpName 
    = SemaRef.Context.DeclarationNames.getCXXOperatorName(Op);
  SemaRef.ArgumentDependentLookup(OpName, Args, NumArgs, Functions);
  
  // Create the overloaded operator invocation for unary operators.
  if (NumArgs == 1 || isPostIncDec) {
    UnaryOperator::Opcode Opc 
      = UnaryOperator::getOverloadedOpcode(Op, isPostIncDec);
    return SemaRef.CreateOverloadedUnaryOp(OpLoc, Opc, Functions, move(First));
  }
  
  // Create the overloaded operator invocation for binary operators.
  BinaryOperator::Opcode Opc = 
    BinaryOperator::getOverloadedOpcode(Op);
  OwningExprResult Result 
    = SemaRef.CreateOverloadedBinOp(OpLoc, Opc, Functions, Args[0], Args[1]);
  if (Result.isInvalid())
    return SemaRef.ExprError();
  
  First.release();
  Second.release();
  return move(Result);  
}
  
} // end namespace clang

#endif // LLVM_CLANG_SEMA_TREETRANSFORM_H
