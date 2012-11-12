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

#include "clang/Basic/SourceLocation.h"

namespace clang {
  class Decl;
  class DeclContext;
  class TagDecl;
  class CXXRecordDecl;
  class ClassTemplateDecl;
  class ClassTemplateSpecializationDecl;
  class FunctionDecl;
  class FunctionTemplateDecl;
  class ObjCCategoryDecl;
  class ObjCInterfaceDecl;
  class ObjCContainerDecl;
  class ObjCPropertyDecl;

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
  virtual void AddedCXXTemplateSpecialization(const FunctionTemplateDecl *TD,
                                              const FunctionDecl *D) {}

  /// \brief An implicit member got a definition.
  virtual void CompletedImplicitDefinition(const FunctionDecl *D) {}

  /// \brief A static data member was implicitly instantiated.
  virtual void StaticDataMemberInstantiated(const VarDecl *D) {}

  /// \brief A new objc category class was added for an interface.
  virtual void AddedObjCCategoryToInterface(const ObjCCategoryDecl *CatD,
                                            const ObjCInterfaceDecl *IFD) {}

  /// \brief A objc class extension redeclared or introduced a property.
  ///
  /// \param Prop the property in the class extension
  ///
  /// \param OrigProp the property from the original interface that was declared
  /// or null if the property was introduced.
  ///
  /// \param ClassExt the class extension.
  virtual void AddedObjCPropertyInClassExtension(const ObjCPropertyDecl *Prop,
                                            const ObjCPropertyDecl *OrigProp,
                                            const ObjCCategoryDecl *ClassExt) {}

  // NOTE: If new methods are added they should also be added to
  // MultiplexASTMutationListener.
};

} // end namespace clang

#endif
