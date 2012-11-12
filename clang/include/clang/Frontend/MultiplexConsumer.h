//===-- MultiplexConsumer.h - AST Consumer for PCH Generation ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file declares the MultiplexConsumer class, which can be used to
//  multiplex ASTConsumer and SemaConsumer messages to many consumers.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_FRONTEND_MULTIPLEXCONSUMER_H
#define CLANG_FRONTEND_MULTIPLEXCONSUMER_H

#include "clang/Basic/LLVM.h"
#include "clang/Sema/SemaConsumer.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/OwningPtr.h"
#include <vector>

namespace clang {

class MultiplexASTMutationListener;
class MultiplexASTDeserializationListener;

// Has a list of ASTConsumers and calls each of them. Owns its children.
class MultiplexConsumer : public SemaConsumer {
public:
  // Takes ownership of the pointers in C.
  MultiplexConsumer(ArrayRef<ASTConsumer*> C);
  ~MultiplexConsumer();

  // ASTConsumer
  virtual void Initialize(ASTContext &Context);
  virtual void HandleCXXStaticMemberVarInstantiation(VarDecl *VD);
  virtual bool HandleTopLevelDecl(DeclGroupRef D);
  virtual void HandleInterestingDecl(DeclGroupRef D);
  virtual void HandleTranslationUnit(ASTContext &Ctx);
  virtual void HandleTagDeclDefinition(TagDecl *D);
  virtual void HandleCXXImplicitFunctionInstantiation(FunctionDecl *D);
  virtual void HandleTopLevelDeclInObjCContainer(DeclGroupRef D);
  virtual void CompleteTentativeDefinition(VarDecl *D);
  virtual void HandleVTable(CXXRecordDecl *RD, bool DefinitionRequired);
  virtual ASTMutationListener *GetASTMutationListener();
  virtual ASTDeserializationListener *GetASTDeserializationListener();
  virtual void PrintStats();

  // SemaConsumer
  virtual void InitializeSema(Sema &S);
  virtual void ForgetSema();

private:
  std::vector<ASTConsumer*> Consumers;  // Owns these.
  OwningPtr<MultiplexASTMutationListener> MutationListener;
  OwningPtr<MultiplexASTDeserializationListener> DeserializationListener;
};

}  // end namespace clang

#endif
