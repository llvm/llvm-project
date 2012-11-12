//===- MultiplexConsumer.cpp - AST Consumer for PCH Generation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the MultiplexConsumer class. It also declares and defines
//  MultiplexASTDeserializationListener and  MultiplexASTMutationListener, which
//  are implementation details of MultiplexConsumer.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/MultiplexConsumer.h"

#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Serialization/ASTDeserializationListener.h"

using namespace clang;

namespace clang {

// This ASTDeserializationListener forwards its notifications to a set of
// child listeners.
class MultiplexASTDeserializationListener
    : public ASTDeserializationListener {
public:
  // Does NOT take ownership of the elements in L.
  MultiplexASTDeserializationListener(
      const std::vector<ASTDeserializationListener*>& L);
  virtual void ReaderInitialized(ASTReader *Reader);
  virtual void IdentifierRead(serialization::IdentID ID,
                              IdentifierInfo *II);
  virtual void TypeRead(serialization::TypeIdx Idx, QualType T);
  virtual void DeclRead(serialization::DeclID ID, const Decl *D);
  virtual void SelectorRead(serialization::SelectorID iD, Selector Sel);
  virtual void MacroDefinitionRead(serialization::PreprocessedEntityID, 
                                   MacroDefinition *MD);
private:
  std::vector<ASTDeserializationListener*> Listeners;
};

MultiplexASTDeserializationListener::MultiplexASTDeserializationListener(
      const std::vector<ASTDeserializationListener*>& L)
    : Listeners(L) {
}

void MultiplexASTDeserializationListener::ReaderInitialized(
    ASTReader *Reader) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->ReaderInitialized(Reader);
}

void MultiplexASTDeserializationListener::IdentifierRead(
    serialization::IdentID ID, IdentifierInfo *II) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->IdentifierRead(ID, II);
}

void MultiplexASTDeserializationListener::TypeRead(
    serialization::TypeIdx Idx, QualType T) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->TypeRead(Idx, T);
}

void MultiplexASTDeserializationListener::DeclRead(
    serialization::DeclID ID, const Decl *D) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->DeclRead(ID, D);
}

void MultiplexASTDeserializationListener::SelectorRead(
    serialization::SelectorID ID, Selector Sel) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->SelectorRead(ID, Sel);
}

void MultiplexASTDeserializationListener::MacroDefinitionRead(
    serialization::PreprocessedEntityID ID, MacroDefinition *MD) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->MacroDefinitionRead(ID, MD);
}

// This ASTMutationListener forwards its notifications to a set of
// child listeners.
class MultiplexASTMutationListener : public ASTMutationListener {
public:
  // Does NOT take ownership of the elements in L.
  MultiplexASTMutationListener(ArrayRef<ASTMutationListener*> L);
  virtual void CompletedTagDefinition(const TagDecl *D);
  virtual void AddedVisibleDecl(const DeclContext *DC, const Decl *D);
  virtual void AddedCXXImplicitMember(const CXXRecordDecl *RD, const Decl *D);
  virtual void AddedCXXTemplateSpecialization(const ClassTemplateDecl *TD,
                                    const ClassTemplateSpecializationDecl *D);
  virtual void AddedCXXTemplateSpecialization(const FunctionTemplateDecl *TD,
                                              const FunctionDecl *D);
  virtual void CompletedImplicitDefinition(const FunctionDecl *D);
  virtual void StaticDataMemberInstantiated(const VarDecl *D);
  virtual void AddedObjCCategoryToInterface(const ObjCCategoryDecl *CatD,
                                            const ObjCInterfaceDecl *IFD);
  virtual void AddedObjCPropertyInClassExtension(const ObjCPropertyDecl *Prop,
                                            const ObjCPropertyDecl *OrigProp,
                                            const ObjCCategoryDecl *ClassExt);
private:
  std::vector<ASTMutationListener*> Listeners;
};

MultiplexASTMutationListener::MultiplexASTMutationListener(
    ArrayRef<ASTMutationListener*> L)
    : Listeners(L.begin(), L.end()) {
}

void MultiplexASTMutationListener::CompletedTagDefinition(const TagDecl *D) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->CompletedTagDefinition(D);
}

void MultiplexASTMutationListener::AddedVisibleDecl(
    const DeclContext *DC, const Decl *D) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->AddedVisibleDecl(DC, D);
}

void MultiplexASTMutationListener::AddedCXXImplicitMember(
    const CXXRecordDecl *RD, const Decl *D) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->AddedCXXImplicitMember(RD, D);
}
void MultiplexASTMutationListener::AddedCXXTemplateSpecialization(
    const ClassTemplateDecl *TD, const ClassTemplateSpecializationDecl *D) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->AddedCXXTemplateSpecialization(TD, D);
}
void MultiplexASTMutationListener::AddedCXXTemplateSpecialization(
    const FunctionTemplateDecl *TD, const FunctionDecl *D) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->AddedCXXTemplateSpecialization(TD, D);
}
void MultiplexASTMutationListener::CompletedImplicitDefinition(
                                                        const FunctionDecl *D) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->CompletedImplicitDefinition(D);
}
void MultiplexASTMutationListener::StaticDataMemberInstantiated(
                                                             const VarDecl *D) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->StaticDataMemberInstantiated(D);
}
void MultiplexASTMutationListener::AddedObjCCategoryToInterface(
                                                 const ObjCCategoryDecl *CatD,
                                                 const ObjCInterfaceDecl *IFD) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->AddedObjCCategoryToInterface(CatD, IFD);
}
void MultiplexASTMutationListener::AddedObjCPropertyInClassExtension(
                                             const ObjCPropertyDecl *Prop,
                                             const ObjCPropertyDecl *OrigProp,
                                             const ObjCCategoryDecl *ClassExt) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->AddedObjCPropertyInClassExtension(Prop, OrigProp, ClassExt);
}

}  // end namespace clang


MultiplexConsumer::MultiplexConsumer(ArrayRef<ASTConsumer*> C)
    : Consumers(C.begin(), C.end()),
      MutationListener(0), DeserializationListener(0) {
  // Collect the mutation listeners and deserialization listeners of all
  // children, and create a multiplex listener each if so.
  std::vector<ASTMutationListener*> mutationListeners;
  std::vector<ASTDeserializationListener*> serializationListeners;
  for (size_t i = 0, e = Consumers.size(); i != e; ++i) {
    ASTMutationListener* mutationListener =
        Consumers[i]->GetASTMutationListener();
    if (mutationListener)
      mutationListeners.push_back(mutationListener);
    ASTDeserializationListener* serializationListener =
        Consumers[i]->GetASTDeserializationListener();
    if (serializationListener)
      serializationListeners.push_back(serializationListener);
  }
  if (mutationListeners.size()) {
    MutationListener.reset(new MultiplexASTMutationListener(mutationListeners));
  }
  if (serializationListeners.size()) {
    DeserializationListener.reset(
        new MultiplexASTDeserializationListener(serializationListeners));
  }
}

MultiplexConsumer::~MultiplexConsumer() {
  for (size_t i = 0, e = Consumers.size(); i != e; ++i)
    delete Consumers[i];
}

void MultiplexConsumer::Initialize(ASTContext &Context) {
  for (size_t i = 0, e = Consumers.size(); i != e; ++i)
    Consumers[i]->Initialize(Context);
}

bool MultiplexConsumer::HandleTopLevelDecl(DeclGroupRef D) {
  bool Continue = true;
  for (size_t i = 0, e = Consumers.size(); i != e; ++i)
    Continue = Continue && Consumers[i]->HandleTopLevelDecl(D);
  return Continue;
}

void  MultiplexConsumer::HandleCXXStaticMemberVarInstantiation(VarDecl *VD) {
  for (size_t i = 0, e = Consumers.size(); i != e; ++i)
    Consumers[i]->HandleCXXStaticMemberVarInstantiation(VD);
}

void MultiplexConsumer::HandleInterestingDecl(DeclGroupRef D) {
  for (size_t i = 0, e = Consumers.size(); i != e; ++i)
    Consumers[i]->HandleInterestingDecl(D);
}

void MultiplexConsumer::HandleTranslationUnit(ASTContext &Ctx) {
  for (size_t i = 0, e = Consumers.size(); i != e; ++i)
    Consumers[i]->HandleTranslationUnit(Ctx);
}

void MultiplexConsumer::HandleTagDeclDefinition(TagDecl *D) {
  for (size_t i = 0, e = Consumers.size(); i != e; ++i)
    Consumers[i]->HandleTagDeclDefinition(D);
}

void MultiplexConsumer::HandleCXXImplicitFunctionInstantiation(FunctionDecl *D){
  for (size_t i = 0, e = Consumers.size(); i != e; ++i)
    Consumers[i]->HandleCXXImplicitFunctionInstantiation(D);
}

void MultiplexConsumer::HandleTopLevelDeclInObjCContainer(DeclGroupRef D) {
  for (size_t i = 0, e = Consumers.size(); i != e; ++i)
    Consumers[i]->HandleTopLevelDeclInObjCContainer(D);
}

void MultiplexConsumer::CompleteTentativeDefinition(VarDecl *D) {
  for (size_t i = 0, e = Consumers.size(); i != e; ++i)
    Consumers[i]->CompleteTentativeDefinition(D);
}

void MultiplexConsumer::HandleVTable(
    CXXRecordDecl *RD, bool DefinitionRequired) {
  for (size_t i = 0, e = Consumers.size(); i != e; ++i)
    Consumers[i]->HandleVTable(RD, DefinitionRequired);
}

ASTMutationListener *MultiplexConsumer::GetASTMutationListener() {
  return MutationListener.get();
}

ASTDeserializationListener *MultiplexConsumer::GetASTDeserializationListener() {
  return DeserializationListener.get();
}

void MultiplexConsumer::PrintStats() {
  for (size_t i = 0, e = Consumers.size(); i != e; ++i)
    Consumers[i]->PrintStats();
}

void MultiplexConsumer::InitializeSema(Sema &S) {
  for (size_t i = 0, e = Consumers.size(); i != e; ++i)
    if (SemaConsumer *SC = dyn_cast<SemaConsumer>(Consumers[i]))
      SC->InitializeSema(S);
}

void MultiplexConsumer::ForgetSema() {
  for (size_t i = 0, e = Consumers.size(); i != e; ++i)
    if (SemaConsumer *SC = dyn_cast<SemaConsumer>(Consumers[i]))
      SC->ForgetSema();
}
