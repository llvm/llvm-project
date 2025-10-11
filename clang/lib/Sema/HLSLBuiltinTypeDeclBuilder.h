//===--- HLSLBuiltinTypeDeclBuilder.h - HLSL Builtin Type Decl Builder  ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper classes for creating HLSL builtin class types. Used by external HLSL
// sema source.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_HLSLBUILTINTYPEDECLBUILDER_H
#define LLVM_CLANG_SEMA_HLSLBUILTINTYPEDECLBUILDER_H

#include "clang/AST/Type.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/StringMap.h"

using llvm::hlsl::ResourceClass;

namespace clang {

class ClassTemplateDecl;
class NamespaceDecl;
class CXXRecordDecl;
class FieldDecl;

namespace hlsl {

// Builder for builtin HLSL class types such as HLSL resource classes.
// Allows creating declaration of builtin types using the builder pattern
// like this:
//
//   Decl = BuiltinTypeDeclBuilder(Sema, Namespace, "BuiltinClassName")
//           .addSimpleTemplateParams({"T"}, Concept)
//           .finalizeForwardDeclaration();
//
// And then completing the type like this:
//
//   BuiltinTypeDeclBuilder(Sema, Decl)
//          .addDefaultHandleConstructor();
//          .addLoadMethods()
//          .completeDefinition();
//
class BuiltinTypeDeclBuilder {
private:
  Sema &SemaRef;
  CXXRecordDecl *Record = nullptr;
  ClassTemplateDecl *Template = nullptr;
  ClassTemplateDecl *PrevTemplate = nullptr;
  NamespaceDecl *HLSLNamespace = nullptr;
  llvm::StringMap<FieldDecl *> Fields;

public:
  friend struct TemplateParameterListBuilder;
  friend struct BuiltinTypeMethodBuilder;

  BuiltinTypeDeclBuilder(Sema &SemaRef, CXXRecordDecl *R);
  BuiltinTypeDeclBuilder(Sema &SemaRef, NamespaceDecl *Namespace,
                         StringRef Name);
  ~BuiltinTypeDeclBuilder();

  BuiltinTypeDeclBuilder &addSimpleTemplateParams(ArrayRef<StringRef> Names,
                                                  ConceptDecl *CD);
  CXXRecordDecl *finalizeForwardDeclaration() { return Record; }
  BuiltinTypeDeclBuilder &completeDefinition();

  BuiltinTypeDeclBuilder &
  addMemberVariable(StringRef Name, QualType Type, llvm::ArrayRef<Attr *> Attrs,
                    AccessSpecifier Access = AccessSpecifier::AS_private);

  BuiltinTypeDeclBuilder &
  addBufferHandles(ResourceClass RC, bool IsROV, bool RawBuffer,
                   bool HasCounter,
                   AccessSpecifier Access = AccessSpecifier::AS_private);
  BuiltinTypeDeclBuilder &addArraySubscriptOperators();

  // Builtin types constructors
  BuiltinTypeDeclBuilder &addDefaultHandleConstructor();
  BuiltinTypeDeclBuilder &addCopyConstructor();
  BuiltinTypeDeclBuilder &addCopyAssignmentOperator();

  // Static create methods
  BuiltinTypeDeclBuilder &addStaticInitializationFunctions(bool HasCounter);

  // Builtin types methods
  BuiltinTypeDeclBuilder &addLoadMethods();
  BuiltinTypeDeclBuilder &addIncrementCounterMethod();
  BuiltinTypeDeclBuilder &addDecrementCounterMethod();
  BuiltinTypeDeclBuilder &addHandleAccessFunction(DeclarationName &Name,
                                                  bool IsConst, bool IsRef);
  BuiltinTypeDeclBuilder &addAppendMethod();
  BuiltinTypeDeclBuilder &addConsumeMethod();

private:
  BuiltinTypeDeclBuilder &addCreateFromBinding();
  BuiltinTypeDeclBuilder &addCreateFromImplicitBinding();
  BuiltinTypeDeclBuilder &addCreateFromBindingWithImplicitCounter();
  BuiltinTypeDeclBuilder &addCreateFromImplicitBindingWithImplicitCounter();
  BuiltinTypeDeclBuilder &addResourceMember(StringRef MemberName,
                                            ResourceClass RC, bool IsROV,
                                            bool RawBuffer, bool IsCounter,
                                            AccessSpecifier Access);
  BuiltinTypeDeclBuilder &
  addHandleMember(ResourceClass RC, bool IsROV, bool RawBuffer,
                  AccessSpecifier Access = AccessSpecifier::AS_private);
  BuiltinTypeDeclBuilder &
  addCounterHandleMember(ResourceClass RC, bool IsROV, bool RawBuffer,
                         AccessSpecifier Access = AccessSpecifier::AS_private);
  FieldDecl *getResourceHandleField() const;
  FieldDecl *getResourceCounterHandleField() const;
  QualType getFirstTemplateTypeParam();
  QualType getHandleElementType();
  Expr *getConstantIntExpr(int value);
  HLSLAttributedResourceType::Attributes getResourceAttrs() const;
};

} // namespace hlsl

} // namespace clang

#endif // LLVM_CLANG_SEMA_HLSLBUILTINTYPEDECLBUILDER_H
