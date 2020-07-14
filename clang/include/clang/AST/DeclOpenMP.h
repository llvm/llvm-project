//===- DeclOpenMP.h - Classes for representing OpenMP directives -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines OpenMP nodes for declarative directives.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLOPENMP_H
#define LLVM_CLANG_AST_DECLOPENMP_H

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TrailingObjects.h"

namespace clang {

/// This represents '#pragma omp threadprivate ...' directive.
/// For example, in the following, both 'a' and 'A::b' are threadprivate:
///
/// \code
/// int a;
/// #pragma omp threadprivate(a)
/// struct A {
///   static int b;
/// #pragma omp threadprivate(b)
/// };
/// \endcode
///
class OMPThreadPrivateDecl final
    : public Decl,
      private llvm::TrailingObjects<OMPThreadPrivateDecl, Expr *> {
  friend class ASTDeclReader;
  friend TrailingObjects;

  unsigned NumVars;

  virtual void anchor();

  OMPThreadPrivateDecl(Kind DK, DeclContext *DC, SourceLocation L) :
    Decl(DK, DC, L), NumVars(0) { }

  ArrayRef<const Expr *> getVars() const {
    return llvm::makeArrayRef(getTrailingObjects<Expr *>(), NumVars);
  }

  MutableArrayRef<Expr *> getVars() {
    return MutableArrayRef<Expr *>(getTrailingObjects<Expr *>(), NumVars);
  }

  void setVars(ArrayRef<Expr *> VL);

public:
  static OMPThreadPrivateDecl *Create(ASTContext &C, DeclContext *DC,
                                      SourceLocation L,
                                      ArrayRef<Expr *> VL);
  static OMPThreadPrivateDecl *CreateDeserialized(ASTContext &C,
                                                  unsigned ID, unsigned N);

  typedef MutableArrayRef<Expr *>::iterator varlist_iterator;
  typedef ArrayRef<const Expr *>::iterator varlist_const_iterator;
  typedef llvm::iterator_range<varlist_iterator> varlist_range;
  typedef llvm::iterator_range<varlist_const_iterator> varlist_const_range;

  unsigned varlist_size() const { return NumVars; }
  bool varlist_empty() const { return NumVars == 0; }

  varlist_range varlists() {
    return varlist_range(varlist_begin(), varlist_end());
  }
  varlist_const_range varlists() const {
    return varlist_const_range(varlist_begin(), varlist_end());
  }
  varlist_iterator varlist_begin() { return getVars().begin(); }
  varlist_iterator varlist_end() { return getVars().end(); }
  varlist_const_iterator varlist_begin() const { return getVars().begin(); }
  varlist_const_iterator varlist_end() const { return getVars().end(); }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == OMPThreadPrivate; }
};

/// This represents '#pragma omp declare reduction ...' directive.
/// For example, in the following, declared reduction 'foo' for types 'int' and
/// 'float':
///
/// \code
/// #pragma omp declare reduction (foo : int,float : omp_out += omp_in) \
///                     initializer (omp_priv = 0)
/// \endcode
///
/// Here 'omp_out += omp_in' is a combiner and 'omp_priv = 0' is an initializer.
class OMPDeclareReductionDecl final : public ValueDecl, public DeclContext {
  // This class stores some data in DeclContext::OMPDeclareReductionDeclBits
  // to save some space. Use the provided accessors to access it.
public:
  enum InitKind {
    CallInit,   // Initialized by function call.
    DirectInit, // omp_priv(<expr>)
    CopyInit    // omp_priv = <expr>
  };

private:
  friend class ASTDeclReader;
  /// Combiner for declare reduction construct.
  Expr *Combiner = nullptr;
  /// Initializer for declare reduction construct.
  Expr *Initializer = nullptr;
  /// In parameter of the combiner.
  Expr *In = nullptr;
  /// Out parameter of the combiner.
  Expr *Out = nullptr;
  /// Priv parameter of the initializer.
  Expr *Priv = nullptr;
  /// Orig parameter of the initializer.
  Expr *Orig = nullptr;

  /// Reference to the previous declare reduction construct in the same
  /// scope with the same name. Required for proper templates instantiation if
  /// the declare reduction construct is declared inside compound statement.
  LazyDeclPtr PrevDeclInScope;

  void anchor() override;

  OMPDeclareReductionDecl(Kind DK, DeclContext *DC, SourceLocation L,
                          DeclarationName Name, QualType Ty,
                          OMPDeclareReductionDecl *PrevDeclInScope);

  void setPrevDeclInScope(OMPDeclareReductionDecl *Prev) {
    PrevDeclInScope = Prev;
  }

public:
  /// Create declare reduction node.
  static OMPDeclareReductionDecl *
  Create(ASTContext &C, DeclContext *DC, SourceLocation L, DeclarationName Name,
         QualType T, OMPDeclareReductionDecl *PrevDeclInScope);
  /// Create deserialized declare reduction node.
  static OMPDeclareReductionDecl *CreateDeserialized(ASTContext &C,
                                                     unsigned ID);

  /// Get combiner expression of the declare reduction construct.
  Expr *getCombiner() { return Combiner; }
  const Expr *getCombiner() const { return Combiner; }
  /// Get In variable of the combiner.
  Expr *getCombinerIn() { return In; }
  const Expr *getCombinerIn() const { return In; }
  /// Get Out variable of the combiner.
  Expr *getCombinerOut() { return Out; }
  const Expr *getCombinerOut() const { return Out; }
  /// Set combiner expression for the declare reduction construct.
  void setCombiner(Expr *E) { Combiner = E; }
  /// Set combiner In and Out vars.
  void setCombinerData(Expr *InE, Expr *OutE) {
    In = InE;
    Out = OutE;
  }

  /// Get initializer expression (if specified) of the declare reduction
  /// construct.
  Expr *getInitializer() { return Initializer; }
  const Expr *getInitializer() const { return Initializer; }
  /// Get initializer kind.
  InitKind getInitializerKind() const {
    return static_cast<InitKind>(OMPDeclareReductionDeclBits.InitializerKind);
  }
  /// Get Orig variable of the initializer.
  Expr *getInitOrig() { return Orig; }
  const Expr *getInitOrig() const { return Orig; }
  /// Get Priv variable of the initializer.
  Expr *getInitPriv() { return Priv; }
  const Expr *getInitPriv() const { return Priv; }
  /// Set initializer expression for the declare reduction construct.
  void setInitializer(Expr *E, InitKind IK) {
    Initializer = E;
    OMPDeclareReductionDeclBits.InitializerKind = IK;
  }
  /// Set initializer Orig and Priv vars.
  void setInitializerData(Expr *OrigE, Expr *PrivE) {
    Orig = OrigE;
    Priv = PrivE;
  }

  /// Get reference to previous declare reduction construct in the same
  /// scope with the same name.
  OMPDeclareReductionDecl *getPrevDeclInScope();
  const OMPDeclareReductionDecl *getPrevDeclInScope() const;

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == OMPDeclareReduction; }
  static DeclContext *castToDeclContext(const OMPDeclareReductionDecl *D) {
    return static_cast<DeclContext *>(const_cast<OMPDeclareReductionDecl *>(D));
  }
  static OMPDeclareReductionDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<OMPDeclareReductionDecl *>(
        const_cast<DeclContext *>(DC));
  }
};

/// This represents '#pragma omp declare mapper ...' directive. Map clauses are
/// allowed to use with this directive. The following example declares a user
/// defined mapper for the type 'struct vec'. This example instructs the fields
/// 'len' and 'data' should be mapped when mapping instances of 'struct vec'.
///
/// \code
/// #pragma omp declare mapper(mid: struct vec v) map(v.len, v.data[0:N])
/// \endcode
class OMPDeclareMapperDecl final : public ValueDecl, public DeclContext {
  friend class ASTDeclReader;

  /// Clauses associated with this mapper declaration
  MutableArrayRef<OMPClause *> Clauses;

  /// Mapper variable, which is 'v' in the example above
  Expr *MapperVarRef = nullptr;

  /// Name of the mapper variable
  DeclarationName VarName;

  LazyDeclPtr PrevDeclInScope;

  void anchor() override;

  OMPDeclareMapperDecl(Kind DK, DeclContext *DC, SourceLocation L,
                       DeclarationName Name, QualType Ty,
                       DeclarationName VarName,
                       OMPDeclareMapperDecl *PrevDeclInScope)
      : ValueDecl(DK, DC, L, Name, Ty), DeclContext(DK), VarName(VarName),
        PrevDeclInScope(PrevDeclInScope) {}

  void setPrevDeclInScope(OMPDeclareMapperDecl *Prev) {
    PrevDeclInScope = Prev;
  }

  /// Sets an array of clauses to this mapper declaration
  void setClauses(ArrayRef<OMPClause *> CL);

public:
  /// Creates declare mapper node.
  static OMPDeclareMapperDecl *Create(ASTContext &C, DeclContext *DC,
                                      SourceLocation L, DeclarationName Name,
                                      QualType T, DeclarationName VarName,
                                      OMPDeclareMapperDecl *PrevDeclInScope);
  /// Creates deserialized declare mapper node.
  static OMPDeclareMapperDecl *CreateDeserialized(ASTContext &C, unsigned ID,
                                                  unsigned N);

  /// Creates an array of clauses to this mapper declaration and intializes
  /// them.
  void CreateClauses(ASTContext &C, ArrayRef<OMPClause *> CL);

  using clauselist_iterator = MutableArrayRef<OMPClause *>::iterator;
  using clauselist_const_iterator = ArrayRef<const OMPClause *>::iterator;
  using clauselist_range = llvm::iterator_range<clauselist_iterator>;
  using clauselist_const_range =
      llvm::iterator_range<clauselist_const_iterator>;

  unsigned clauselist_size() const { return Clauses.size(); }
  bool clauselist_empty() const { return Clauses.empty(); }

  clauselist_range clauselists() {
    return clauselist_range(clauselist_begin(), clauselist_end());
  }
  clauselist_const_range clauselists() const {
    return clauselist_const_range(clauselist_begin(), clauselist_end());
  }
  clauselist_iterator clauselist_begin() { return Clauses.begin(); }
  clauselist_iterator clauselist_end() { return Clauses.end(); }
  clauselist_const_iterator clauselist_begin() const { return Clauses.begin(); }
  clauselist_const_iterator clauselist_end() const { return Clauses.end(); }

  /// Get the variable declared in the mapper
  Expr *getMapperVarRef() { return MapperVarRef; }
  const Expr *getMapperVarRef() const { return MapperVarRef; }
  /// Set the variable declared in the mapper
  void setMapperVarRef(Expr *MapperVarRefE) { MapperVarRef = MapperVarRefE; }

  /// Get the name of the variable declared in the mapper
  DeclarationName getVarName() { return VarName; }

  /// Get reference to previous declare mapper construct in the same
  /// scope with the same name.
  OMPDeclareMapperDecl *getPrevDeclInScope();
  const OMPDeclareMapperDecl *getPrevDeclInScope() const;

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == OMPDeclareMapper; }
  static DeclContext *castToDeclContext(const OMPDeclareMapperDecl *D) {
    return static_cast<DeclContext *>(const_cast<OMPDeclareMapperDecl *>(D));
  }
  static OMPDeclareMapperDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<OMPDeclareMapperDecl *>(const_cast<DeclContext *>(DC));
  }
};

/// Pseudo declaration for capturing expressions. Also is used for capturing of
/// non-static data members in non-static member functions.
///
/// Clang supports capturing of variables only, but OpenMP 4.5 allows to
/// privatize non-static members of current class in non-static member
/// functions. This pseudo-declaration allows properly handle this kind of
/// capture by wrapping captured expression into a variable-like declaration.
class OMPCapturedExprDecl final : public VarDecl {
  friend class ASTDeclReader;
  void anchor() override;

  OMPCapturedExprDecl(ASTContext &C, DeclContext *DC, IdentifierInfo *Id,
                      QualType Type, TypeSourceInfo *TInfo,
                      SourceLocation StartLoc)
      : VarDecl(OMPCapturedExpr, C, DC, StartLoc, StartLoc, Id, Type, TInfo,
                SC_None) {
    setImplicit();
  }

public:
  static OMPCapturedExprDecl *Create(ASTContext &C, DeclContext *DC,
                                     IdentifierInfo *Id, QualType T,
                                     SourceLocation StartLoc);

  static OMPCapturedExprDecl *CreateDeserialized(ASTContext &C, unsigned ID);

  SourceRange getSourceRange() const override LLVM_READONLY;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == OMPCapturedExpr; }
};

/// This represents '#pragma omp requires...' directive.
/// For example
///
/// \code
/// #pragma omp requires unified_address
/// \endcode
///
class OMPRequiresDecl final
    : public Decl,
      private llvm::TrailingObjects<OMPRequiresDecl, OMPClause *> {
  friend class ASTDeclReader;
  friend TrailingObjects;

  // Number of clauses associated with this requires declaration
  unsigned NumClauses = 0;

  virtual void anchor();

  OMPRequiresDecl(Kind DK, DeclContext *DC, SourceLocation L)
      : Decl(DK, DC, L), NumClauses(0) {}

  /// Returns an array of immutable clauses associated with this requires
  /// declaration
  ArrayRef<const OMPClause *> getClauses() const {
    return llvm::makeArrayRef(getTrailingObjects<OMPClause *>(), NumClauses);
  }

  /// Returns an array of clauses associated with this requires declaration
  MutableArrayRef<OMPClause *> getClauses() {
    return MutableArrayRef<OMPClause *>(getTrailingObjects<OMPClause *>(),
                                        NumClauses);
  }

  /// Sets an array of clauses to this requires declaration
  void setClauses(ArrayRef<OMPClause *> CL);

public:
  /// Create requires node.
  static OMPRequiresDecl *Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation L, ArrayRef<OMPClause *> CL);
  /// Create deserialized requires node.
  static OMPRequiresDecl *CreateDeserialized(ASTContext &C, unsigned ID,
                                             unsigned N);

  using clauselist_iterator = MutableArrayRef<OMPClause *>::iterator;
  using clauselist_const_iterator = ArrayRef<const OMPClause *>::iterator;
  using clauselist_range = llvm::iterator_range<clauselist_iterator>;
  using clauselist_const_range = llvm::iterator_range<clauselist_const_iterator>;

  unsigned clauselist_size() const { return NumClauses; }
  bool clauselist_empty() const { return NumClauses == 0; }

  clauselist_range clauselists() {
    return clauselist_range(clauselist_begin(), clauselist_end());
  }
  clauselist_const_range clauselists() const {
    return clauselist_const_range(clauselist_begin(), clauselist_end());
  }
  clauselist_iterator clauselist_begin() { return getClauses().begin(); }
  clauselist_iterator clauselist_end() { return getClauses().end(); }
  clauselist_const_iterator clauselist_begin() const {
    return getClauses().begin();
  }
  clauselist_const_iterator clauselist_end() const {
    return getClauses().end();
  }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == OMPRequires; }
};

/// This represents '#pragma omp allocate ...' directive.
/// For example, in the following, the default allocator is used for both 'a'
/// and 'A::b':
///
/// \code
/// int a;
/// #pragma omp allocate(a)
/// struct A {
///   static int b;
/// #pragma omp allocate(b)
/// };
/// \endcode
///
class OMPAllocateDecl final
    : public Decl,
      private llvm::TrailingObjects<OMPAllocateDecl, Expr *, OMPClause *> {
  friend class ASTDeclReader;
  friend TrailingObjects;

  /// Number of variable within the allocate directive.
  unsigned NumVars = 0;
  /// Number of clauses associated with the allocate directive.
  unsigned NumClauses = 0;

  size_t numTrailingObjects(OverloadToken<Expr *>) const {
    return NumVars;
  }
  size_t numTrailingObjects(OverloadToken<OMPClause *>) const {
    return NumClauses;
  }

  virtual void anchor();

  OMPAllocateDecl(Kind DK, DeclContext *DC, SourceLocation L)
      : Decl(DK, DC, L) {}

  ArrayRef<const Expr *> getVars() const {
    return llvm::makeArrayRef(getTrailingObjects<Expr *>(), NumVars);
  }

  MutableArrayRef<Expr *> getVars() {
    return MutableArrayRef<Expr *>(getTrailingObjects<Expr *>(), NumVars);
  }

  void setVars(ArrayRef<Expr *> VL);

  /// Returns an array of immutable clauses associated with this directive.
  ArrayRef<OMPClause *> getClauses() const {
    return llvm::makeArrayRef(getTrailingObjects<OMPClause *>(), NumClauses);
  }

  /// Returns an array of clauses associated with this directive.
  MutableArrayRef<OMPClause *> getClauses() {
    return MutableArrayRef<OMPClause *>(getTrailingObjects<OMPClause *>(),
                                        NumClauses);
  }

  /// Sets an array of clauses to this requires declaration
  void setClauses(ArrayRef<OMPClause *> CL);

public:
  static OMPAllocateDecl *Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation L, ArrayRef<Expr *> VL,
                                 ArrayRef<OMPClause *> CL);
  static OMPAllocateDecl *CreateDeserialized(ASTContext &C, unsigned ID,
                                             unsigned NVars, unsigned NClauses);

  typedef MutableArrayRef<Expr *>::iterator varlist_iterator;
  typedef ArrayRef<const Expr *>::iterator varlist_const_iterator;
  typedef llvm::iterator_range<varlist_iterator> varlist_range;
  typedef llvm::iterator_range<varlist_const_iterator> varlist_const_range;
  using clauselist_iterator = MutableArrayRef<OMPClause *>::iterator;
  using clauselist_const_iterator = ArrayRef<const OMPClause *>::iterator;
  using clauselist_range = llvm::iterator_range<clauselist_iterator>;
  using clauselist_const_range = llvm::iterator_range<clauselist_const_iterator>;


  unsigned varlist_size() const { return NumVars; }
  bool varlist_empty() const { return NumVars == 0; }
  unsigned clauselist_size() const { return NumClauses; }
  bool clauselist_empty() const { return NumClauses == 0; }

  varlist_range varlists() {
    return varlist_range(varlist_begin(), varlist_end());
  }
  varlist_const_range varlists() const {
    return varlist_const_range(varlist_begin(), varlist_end());
  }
  varlist_iterator varlist_begin() { return getVars().begin(); }
  varlist_iterator varlist_end() { return getVars().end(); }
  varlist_const_iterator varlist_begin() const { return getVars().begin(); }
  varlist_const_iterator varlist_end() const { return getVars().end(); }

  clauselist_range clauselists() {
    return clauselist_range(clauselist_begin(), clauselist_end());
  }
  clauselist_const_range clauselists() const {
    return clauselist_const_range(clauselist_begin(), clauselist_end());
  }
  clauselist_iterator clauselist_begin() { return getClauses().begin(); }
  clauselist_iterator clauselist_end() { return getClauses().end(); }
  clauselist_const_iterator clauselist_begin() const {
    return getClauses().begin();
  }
  clauselist_const_iterator clauselist_end() const {
    return getClauses().end();
  }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == OMPAllocate; }
};

} // end namespace clang

#endif
