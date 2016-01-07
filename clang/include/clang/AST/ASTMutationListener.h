//===--- ASTMutationListener.h - AST Mutation Interface --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTMutationListener interface.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_ASTMUTATIONLISTENER_H
#define LLVM_CLANG_AST_ASTMUTATIONLISTENER_H

namespace clang {
  class Attr;
  class ClassTemplateDecl;
  class ClassTemplateSpecializationDecl;
  class CXXDestructorDecl;
  class CXXRecordDecl;
  class Decl;
  class DeclContext;
  class FunctionDecl;
  class FunctionTemplateDecl;
  class Module;
  class NamedDecl;
  class ObjCCategoryDecl;
  class ObjCContainerDecl;
  class ObjCInterfaceDecl;
  class ObjCPropertyDecl;
  class ParmVarDecl;
  class QualType;
  class RecordDecl;
  class TagDecl;
  class VarDecl;
  class VarTemplateDecl;
  class VarTemplateSpecializationDecl;

/// \brief An abstract interface that should be implemented by listeners
/// that want to be notified when an AST entity gets modified after its
/// initial creation.
class ASTMutationListener {
public:
  virtual ~ASTMutationListener();

  /// \brief A new TagDecl definition was completed.
  virtual void CompletedTagDefinition(const TagDecl *D) { }

  /// \brief A new declaration with name has been added to a DeclContext.
  virtual void AddedVisibleDecl(const DeclContext *DC, const Decl *D) {}

  /// \brief An implicit member was added after the definition was completed.
  virtual void AddedCXXImplicitMember(const CXXRecordDecl *RD, const Decl *D) {}

  /// \brief A template specialization (or partial one) was added to the
  /// template declaration.
  virtual void AddedCXXTemplateSpecialization(const ClassTemplateDecl *TD,
                                    const ClassTemplateSpecializationDecl *D) {}

  /// \brief A template specialization (or partial one) was added to the
  /// template declaration.
  virtual void
  AddedCXXTemplateSpecialization(const VarTemplateDecl *TD,
                                 const VarTemplateSpecializationDecl *D) {}

  /// \brief A template specialization (or partial one) was added to the
  /// template declaration.
  virtual void AddedCXXTemplateSpecialization(const FunctionTemplateDecl *TD,
                                              const FunctionDecl *D) {}

  /// \brief A function's exception specification has been evaluated or
  /// instantiated.
  virtual void ResolvedExceptionSpec(const FunctionDecl *FD) {}

  /// \brief A function's return type has been deduced.
  virtual void DeducedReturnType(const FunctionDecl *FD, QualType ReturnType);

  /// \brief A virtual destructor's operator delete has been resolved.
  virtual void ResolvedOperatorDelete(const CXXDestructorDecl *DD,
                                      const FunctionDecl *Delete) {}

  /// \brief An implicit member got a definition.
  virtual void CompletedImplicitDefinition(const FunctionDecl *D) {}

  /// \brief A static data member was implicitly instantiated.
  virtual void StaticDataMemberInstantiated(const VarDecl *D) {}

  /// \brief A function template's definition was instantiated.
  virtual void FunctionDefinitionInstantiated(const FunctionDecl *D) {}

  /// \brief A default argument was instantiated.
  virtual void DefaultArgumentInstantiated(const ParmVarDecl *D) {}

  /// \brief A new objc category class was added for an interface.
  virtual void AddedObjCCategoryToInterface(const ObjCCategoryDecl *CatD,
                                            const ObjCInterfaceDecl *IFD) {}

  /// \brief A declaration is marked used which was not previously marked used.
  ///
  /// \param D the declaration marked used
  virtual void DeclarationMarkedUsed(const Decl *D) {}

  /// \brief A declaration is marked as OpenMP threadprivate which was not
  /// previously marked as threadprivate.
  ///
  /// \param D the declaration marked OpenMP threadprivate.
  virtual void DeclarationMarkedOpenMPThreadPrivate(const Decl *D) {}

  /// \brief A definition has been made visible by being redefined locally.
  ///
  /// \param D The definition that was previously not visible.
  /// \param M The containing module in which the definition was made visible,
  ///        if any.
  virtual void RedefinedHiddenDefinition(const NamedDecl *D, Module *M) {}
  
  /// \brief An attribute was added to a RecordDecl
  ///
  /// \param Attr The attribute that was added to the Record
  ///
  /// \param Record The RecordDecl that got a new attribute
  virtual void AddedAttributeToRecord(const Attr *Attr, 
                                      const RecordDecl *Record) {}

  // NOTE: If new methods are added they should also be added to
  // MultiplexASTMutationListener.
};

} // end namespace clang

#endif
