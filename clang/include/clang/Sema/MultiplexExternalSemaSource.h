//===--- MultiplexExternalSemaSource.h - External Sema Interface-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines ExternalSemaSource interface, dispatching to all clients
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SEMA_MULTIPLEX_EXTERNAL_SEMA_SOURCE_H
#define LLVM_CLANG_SEMA_MULTIPLEX_EXTERNAL_SEMA_SOURCE_H

#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/Weak.h"

#include "llvm/ADT/SmallVector.h"

#include <utility>

namespace clang {

  class CXXConstructorDecl;
  class CXXRecordDecl;
  class DeclaratorDecl;
  struct ExternalVTableUse;
  class LookupResult;
  class NamespaceDecl;
  class Scope;
  class Sema;
  class TypedefNameDecl;
  class ValueDecl;
  class VarDecl;


/// \brief An abstract interface that should be implemented by
/// external AST sources that also provide information for semantic
/// analysis.
class MultiplexExternalSemaSource : public ExternalSemaSource {

private:
  llvm::SmallVector<ExternalSemaSource*, 2> Sources; // doesn't own them.

public:
  
  ///\brief Constructs a new multiplexing external sema source and appends the
  /// given element to it.
  ///
  ///\param[in] s1 - A non-null (old) ExternalSemaSource.
  ///\param[in] s2 - A non-null (new) ExternalSemaSource.
  ///
  MultiplexExternalSemaSource(ExternalSemaSource& s1, ExternalSemaSource& s2);

  ~MultiplexExternalSemaSource();

  ///\brief Appends new source to the source list.
  ///
  ///\param[in] source - An ExternalSemaSource.
  ///
  void addSource(ExternalSemaSource &source);

  //===--------------------------------------------------------------------===//
  // ExternalASTSource.
  //===--------------------------------------------------------------------===//

  /// \brief Resolve a declaration ID into a declaration, potentially
  /// building a new declaration.
  ///
  /// This method only needs to be implemented if the AST source ever
  /// passes back decl sets as VisibleDeclaration objects.
  ///
  /// The default implementation of this method is a no-op.
  virtual Decl *GetExternalDecl(uint32_t ID);

  /// \brief Resolve a selector ID into a selector.
  ///
  /// This operation only needs to be implemented if the AST source
  /// returns non-zero for GetNumKnownSelectors().
  ///
  /// The default implementation of this method is a no-op.
  virtual Selector GetExternalSelector(uint32_t ID);

  /// \brief Returns the number of selectors known to the external AST
  /// source.
  ///
  /// The default implementation of this method is a no-op.
  virtual uint32_t GetNumExternalSelectors();

  /// \brief Resolve the offset of a statement in the decl stream into
  /// a statement.
  ///
  /// This operation is meant to be used via a LazyOffsetPtr.  It only
  /// needs to be implemented if the AST source uses methods like
  /// FunctionDecl::setLazyBody when building decls.
  ///
  /// The default implementation of this method is a no-op.
  virtual Stmt *GetExternalDeclStmt(uint64_t Offset);

  /// \brief Resolve the offset of a set of C++ base specifiers in the decl
  /// stream into an array of specifiers.
  ///
  /// The default implementation of this method is a no-op.
  virtual CXXBaseSpecifier *GetExternalCXXBaseSpecifiers(uint64_t Offset);

  /// \brief Finds all declarations with the given name in the
  /// given context.
  ///
  /// Generally the final step of this method is either to call
  /// SetExternalVisibleDeclsForName or to recursively call lookup on
  /// the DeclContext after calling SetExternalVisibleDecls.
  ///
  /// The default implementation of this method is a no-op.
  virtual DeclContextLookupResult
  FindExternalVisibleDeclsByName(const DeclContext *DC, DeclarationName Name);

  /// \brief Ensures that the table of all visible declarations inside this
  /// context is up to date.
  ///
  /// The default implementation of this functino is a no-op.
  virtual void completeVisibleDeclsMap(const DeclContext *DC);

  /// \brief Finds all declarations lexically contained within the given
  /// DeclContext, after applying an optional filter predicate.
  ///
  /// \param isKindWeWant a predicate function that returns true if the passed
  /// declaration kind is one we are looking for. If NULL, all declarations
  /// are returned.
  ///
  /// \return an indication of whether the load succeeded or failed.
  ///
  /// The default implementation of this method is a no-op.
  virtual ExternalLoadResult FindExternalLexicalDecls(const DeclContext *DC,
                                        bool (*isKindWeWant)(Decl::Kind),
                                        SmallVectorImpl<Decl*> &Result);

  /// \brief Finds all declarations lexically contained within the given
  /// DeclContext.
  ///
  /// \return true if an error occurred
  ExternalLoadResult FindExternalLexicalDecls(const DeclContext *DC,
                                SmallVectorImpl<Decl*> &Result) {
    return FindExternalLexicalDecls(DC, 0, Result);
  }

  template <typename DeclTy>
  ExternalLoadResult FindExternalLexicalDeclsBy(const DeclContext *DC,
                                  SmallVectorImpl<Decl*> &Result) {
    return FindExternalLexicalDecls(DC, DeclTy::classofKind, Result);
  }

  /// \brief Get the decls that are contained in a file in the Offset/Length
  /// range. \p Length can be 0 to indicate a point at \p Offset instead of
  /// a range. 
  virtual void FindFileRegionDecls(FileID File, unsigned Offset,unsigned Length,
                                   SmallVectorImpl<Decl *> &Decls);

  /// \brief Gives the external AST source an opportunity to complete
  /// an incomplete type.
  virtual void CompleteType(TagDecl *Tag);

  /// \brief Gives the external AST source an opportunity to complete an
  /// incomplete Objective-C class.
  ///
  /// This routine will only be invoked if the "externally completed" bit is
  /// set on the ObjCInterfaceDecl via the function 
  /// \c ObjCInterfaceDecl::setExternallyCompleted().
  virtual void CompleteType(ObjCInterfaceDecl *Class);

  /// \brief Loads comment ranges.
  virtual void ReadComments();

  /// \brief Notify ExternalASTSource that we started deserialization of
  /// a decl or type so until FinishedDeserializing is called there may be
  /// decls that are initializing. Must be paired with FinishedDeserializing.
  ///
  /// The default implementation of this method is a no-op.
  virtual void StartedDeserializing();

  /// \brief Notify ExternalASTSource that we finished the deserialization of
  /// a decl or type. Must be paired with StartedDeserializing.
  ///
  /// The default implementation of this method is a no-op.
  virtual void FinishedDeserializing();

  /// \brief Function that will be invoked when we begin parsing a new
  /// translation unit involving this external AST source.
  ///
  /// The default implementation of this method is a no-op.
  virtual void StartTranslationUnit(ASTConsumer *Consumer);

  /// \brief Print any statistics that have been gathered regarding
  /// the external AST source.
  ///
  /// The default implementation of this method is a no-op.
  virtual void PrintStats();
  
  
  /// \brief Perform layout on the given record.
  ///
  /// This routine allows the external AST source to provide an specific 
  /// layout for a record, overriding the layout that would normally be
  /// constructed. It is intended for clients who receive specific layout
  /// details rather than source code (such as LLDB). The client is expected
  /// to fill in the field offsets, base offsets, virtual base offsets, and
  /// complete object size.
  ///
  /// \param Record The record whose layout is being requested.
  ///
  /// \param Size The final size of the record, in bits.
  ///
  /// \param Alignment The final alignment of the record, in bits.
  ///
  /// \param FieldOffsets The offset of each of the fields within the record,
  /// expressed in bits. All of the fields must be provided with offsets.
  ///
  /// \param BaseOffsets The offset of each of the direct, non-virtual base
  /// classes. If any bases are not given offsets, the bases will be laid 
  /// out according to the ABI.
  ///
  /// \param VirtualBaseOffsets The offset of each of the virtual base classes
  /// (either direct or not). If any bases are not given offsets, the bases will 
  /// be laid out according to the ABI.
  /// 
  /// \returns true if the record layout was provided, false otherwise.
  virtual bool 
  layoutRecordType(const RecordDecl *Record,
                   uint64_t &Size, uint64_t &Alignment,
                   llvm::DenseMap<const FieldDecl *, uint64_t> &FieldOffsets,
                 llvm::DenseMap<const CXXRecordDecl *, CharUnits> &BaseOffsets,
          llvm::DenseMap<const CXXRecordDecl *, CharUnits> &VirtualBaseOffsets);

  /// Return the amount of memory used by memory buffers, breaking down
  /// by heap-backed versus mmap'ed memory.
  virtual void getMemoryBufferSizes(MemoryBufferSizes &sizes) const;

  //===--------------------------------------------------------------------===//
  // ExternalSemaSource.
  //===--------------------------------------------------------------------===//

  /// \brief Initialize the semantic source with the Sema instance
  /// being used to perform semantic analysis on the abstract syntax
  /// tree.
  virtual void InitializeSema(Sema &S);

  /// \brief Inform the semantic consumer that Sema is no longer available.
  virtual void ForgetSema();

  /// \brief Load the contents of the global method pool for a given
  /// selector.
  virtual void ReadMethodPool(Selector Sel);

  /// \brief Load the set of namespaces that are known to the external source,
  /// which will be used during typo correction.
  virtual void ReadKnownNamespaces(SmallVectorImpl<NamespaceDecl*> &Namespaces);
  
  /// \brief Do last resort, unqualified lookup on a LookupResult that
  /// Sema cannot find.
  ///
  /// \param R a LookupResult that is being recovered.
  ///
  /// \param S the Scope of the identifier occurrence.
  ///
  /// \return true to tell Sema to recover using the LookupResult.
  virtual bool LookupUnqualified(LookupResult &R, Scope *S);

  /// \brief Read the set of tentative definitions known to the external Sema
  /// source.
  ///
  /// The external source should append its own tentative definitions to the
  /// given vector of tentative definitions. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  virtual void ReadTentativeDefinitions(SmallVectorImpl<VarDecl*> &Defs);
  
  /// \brief Read the set of unused file-scope declarations known to the
  /// external Sema source.
  ///
  /// The external source should append its own unused, filed-scope to the
  /// given vector of declarations. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  virtual void ReadUnusedFileScopedDecls(
                                 SmallVectorImpl<const DeclaratorDecl*> &Decls);
  
  /// \brief Read the set of delegating constructors known to the
  /// external Sema source.
  ///
  /// The external source should append its own delegating constructors to the
  /// given vector of declarations. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  virtual void ReadDelegatingConstructors(
                                   SmallVectorImpl<CXXConstructorDecl*> &Decls);

  /// \brief Read the set of ext_vector type declarations known to the
  /// external Sema source.
  ///
  /// The external source should append its own ext_vector type declarations to
  /// the given vector of declarations. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  virtual void ReadExtVectorDecls(SmallVectorImpl<TypedefNameDecl*> &Decls);

  /// \brief Read the set of dynamic classes known to the external Sema source.
  ///
  /// The external source should append its own dynamic classes to
  /// the given vector of declarations. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  virtual void ReadDynamicClasses(SmallVectorImpl<CXXRecordDecl*> &Decls);

  /// \brief Read the set of locally-scoped external declarations known to the
  /// external Sema source.
  ///
  /// The external source should append its own locally-scoped external
  /// declarations to the given vector of declarations. Note that this routine 
  /// may be invoked multiple times; the external source should take care not 
  /// to introduce the same declarations repeatedly.
  virtual void ReadLocallyScopedExternalDecls(SmallVectorImpl<NamedDecl*>&Decls);

  /// \brief Read the set of referenced selectors known to the
  /// external Sema source.
  ///
  /// The external source should append its own referenced selectors to the 
  /// given vector of selectors. Note that this routine 
  /// may be invoked multiple times; the external source should take care not 
  /// to introduce the same selectors repeatedly.
  virtual void ReadReferencedSelectors(SmallVectorImpl<std::pair<Selector, 
                                                       SourceLocation> > &Sels);

  /// \brief Read the set of weak, undeclared identifiers known to the
  /// external Sema source.
  ///
  /// The external source should append its own weak, undeclared identifiers to
  /// the given vector. Note that this routine may be invoked multiple times; 
  /// the external source should take care not to introduce the same identifiers
  /// repeatedly.
  virtual void ReadWeakUndeclaredIdentifiers(
                    SmallVectorImpl<std::pair<IdentifierInfo*, WeakInfo> > &WI);

  /// \brief Read the set of used vtables known to the external Sema source.
  ///
  /// The external source should append its own used vtables to the given
  /// vector. Note that this routine may be invoked multiple times; the external
  /// source should take care not to introduce the same vtables repeatedly.
  virtual void ReadUsedVTables(SmallVectorImpl<ExternalVTableUse> &VTables);

  /// \brief Read the set of pending instantiations known to the external
  /// Sema source.
  ///
  /// The external source should append its own pending instantiations to the
  /// given vector. Note that this routine may be invoked multiple times; the
  /// external source should take care not to introduce the same instantiations
  /// repeatedly.
  virtual void ReadPendingInstantiations(
              SmallVectorImpl<std::pair<ValueDecl*, SourceLocation> >& Pending);

  // isa/cast/dyn_cast support
  static bool classof(const MultiplexExternalSemaSource*) { return true; }
  //static bool classof(const ExternalSemaSource*) { return true; }
}; 

} // end namespace clang

#endif // LLVM_CLANG_SEMA_MULTIPLEX_EXTERNAL_SEMA_SOURCE_H
