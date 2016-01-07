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
  void ReaderInitialized(ASTReader *Reader) override;
  void IdentifierRead(serialization::IdentID ID,
                      IdentifierInfo *II) override;
  void MacroRead(serialization::MacroID ID, MacroInfo *MI) override;
  void TypeRead(serialization::TypeIdx Idx, QualType T) override;
  void DeclRead(serialization::DeclID ID, const Decl *D) override;
  void SelectorRead(serialization::SelectorID iD, Selector Sel) override;
  void MacroDefinitionRead(serialization::PreprocessedEntityID,
                           MacroDefinitionRecord *MD) override;
  void ModuleRead(serialization::SubmoduleID ID, Module *Mod) override;

private:
  std::vector<ASTDeserializationListener *> Listeners;
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

void MultiplexASTDeserializationListener::MacroRead(
    serialization::MacroID ID, MacroInfo *MI) {
  for (auto &Listener : Listeners)
    Listener->MacroRead(ID, MI);
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
    serialization::PreprocessedEntityID ID, MacroDefinitionRecord *MD) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->MacroDefinitionRead(ID, MD);
}

void MultiplexASTDeserializationListener::ModuleRead(
    serialization::SubmoduleID ID, Module *Mod) {
  for (auto &Listener : Listeners)
    Listener->ModuleRead(ID, Mod);
}

// This ASTMutationListener forwards its notifications to a set of
// child listeners.
class MultiplexASTMutationListener : public ASTMutationListener {
public:
  // Does NOT take ownership of the elements in L.
  MultiplexASTMutationListener(ArrayRef<ASTMutationListener*> L);
  void CompletedTagDefinition(const TagDecl *D) override;
  void AddedVisibleDecl(const DeclContext *DC, const Decl *D) override;
  void AddedCXXImplicitMember(const CXXRecordDecl *RD, const Decl *D) override;
  void AddedCXXTemplateSpecialization(const ClassTemplateDecl *TD,
                            const ClassTemplateSpecializationDecl *D) override;
  void AddedCXXTemplateSpecialization(const VarTemplateDecl *TD,
                               const VarTemplateSpecializationDecl *D) override;
  void AddedCXXTemplateSpecialization(const FunctionTemplateDecl *TD,
                                      const FunctionDecl *D) override;
  void ResolvedExceptionSpec(const FunctionDecl *FD) override;
  void DeducedReturnType(const FunctionDecl *FD, QualType ReturnType) override;
  void ResolvedOperatorDelete(const CXXDestructorDecl *DD,
                              const FunctionDecl *Delete) override;
  void CompletedImplicitDefinition(const FunctionDecl *D) override;
  void StaticDataMemberInstantiated(const VarDecl *D) override;
  void DefaultArgumentInstantiated(const ParmVarDecl *D) override;
  void AddedObjCCategoryToInterface(const ObjCCategoryDecl *CatD,
                                    const ObjCInterfaceDecl *IFD) override;
  void FunctionDefinitionInstantiated(const FunctionDecl *D) override;
  void DeclarationMarkedUsed(const Decl *D) override;
  void DeclarationMarkedOpenMPThreadPrivate(const Decl *D) override;
  void RedefinedHiddenDefinition(const NamedDecl *D, Module *M) override;
  void AddedAttributeToRecord(const Attr *Attr, 
                              const RecordDecl *Record) override;

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
    const VarTemplateDecl *TD, const VarTemplateSpecializationDecl *D) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->AddedCXXTemplateSpecialization(TD, D);
}
void MultiplexASTMutationListener::AddedCXXTemplateSpecialization(
    const FunctionTemplateDecl *TD, const FunctionDecl *D) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->AddedCXXTemplateSpecialization(TD, D);
}
void MultiplexASTMutationListener::ResolvedExceptionSpec(
    const FunctionDecl *FD) {
  for (auto &Listener : Listeners)
    Listener->ResolvedExceptionSpec(FD);
}
void MultiplexASTMutationListener::DeducedReturnType(const FunctionDecl *FD,
                                                     QualType ReturnType) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->DeducedReturnType(FD, ReturnType);
}
void MultiplexASTMutationListener::ResolvedOperatorDelete(
    const CXXDestructorDecl *DD, const FunctionDecl *Delete) {
  for (auto *L : Listeners)
    L->ResolvedOperatorDelete(DD, Delete);
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
void MultiplexASTMutationListener::DefaultArgumentInstantiated(
                                                         const ParmVarDecl *D) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->DefaultArgumentInstantiated(D);
}
void MultiplexASTMutationListener::AddedObjCCategoryToInterface(
                                                 const ObjCCategoryDecl *CatD,
                                                 const ObjCInterfaceDecl *IFD) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->AddedObjCCategoryToInterface(CatD, IFD);
}
void MultiplexASTMutationListener::FunctionDefinitionInstantiated(
    const FunctionDecl *D) {
  for (auto &Listener : Listeners)
    Listener->FunctionDefinitionInstantiated(D);
}
void MultiplexASTMutationListener::DeclarationMarkedUsed(const Decl *D) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->DeclarationMarkedUsed(D);
}
void MultiplexASTMutationListener::DeclarationMarkedOpenMPThreadPrivate(
    const Decl *D) {
  for (size_t i = 0, e = Listeners.size(); i != e; ++i)
    Listeners[i]->DeclarationMarkedOpenMPThreadPrivate(D);
}
void MultiplexASTMutationListener::RedefinedHiddenDefinition(const NamedDecl *D,
                                                             Module *M) {
  for (auto *L : Listeners)
    L->RedefinedHiddenDefinition(D, M);
}
  
void MultiplexASTMutationListener::AddedAttributeToRecord(
                                                    const Attr *Attr, 
                                                    const RecordDecl *Record) {
  for (auto *L : Listeners)
    L->AddedAttributeToRecord(Attr, Record);
}

}  // end namespace clang

MultiplexConsumer::MultiplexConsumer(
    std::vector<std::unique_ptr<ASTConsumer>> C)
    : Consumers(std::move(C)), MutationListener(), DeserializationListener() {
  // Collect the mutation listeners and deserialization listeners of all
  // children, and create a multiplex listener each if so.
  std::vector<ASTMutationListener*> mutationListeners;
  std::vector<ASTDeserializationListener*> serializationListeners;
  for (auto &Consumer : Consumers) {
    if (auto *mutationListener = Consumer->GetASTMutationListener())
      mutationListeners.push_back(mutationListener);
    if (auto *serializationListener = Consumer->GetASTDeserializationListener())
      serializationListeners.push_back(serializationListener);
  }
  if (!mutationListeners.empty()) {
    MutationListener =
        llvm::make_unique<MultiplexASTMutationListener>(mutationListeners);
  }
  if (!serializationListeners.empty()) {
    DeserializationListener =
        llvm::make_unique<MultiplexASTDeserializationListener>(
            serializationListeners);
  }
}

MultiplexConsumer::~MultiplexConsumer() {}

void MultiplexConsumer::Initialize(ASTContext &Context) {
  for (auto &Consumer : Consumers)
    Consumer->Initialize(Context);
}

bool MultiplexConsumer::HandleTopLevelDecl(DeclGroupRef D) {
  bool Continue = true;
  for (auto &Consumer : Consumers)
    Continue = Continue && Consumer->HandleTopLevelDecl(D);
  return Continue;
}

void MultiplexConsumer::HandleInlineMethodDefinition(CXXMethodDecl *D) {
  for (auto &Consumer : Consumers)
    Consumer->HandleInlineMethodDefinition(D);
}

void MultiplexConsumer::HandleCXXStaticMemberVarInstantiation(VarDecl *VD) {
  for (auto &Consumer : Consumers)
    Consumer->HandleCXXStaticMemberVarInstantiation(VD);
}

void MultiplexConsumer::HandleInterestingDecl(DeclGroupRef D) {
  for (auto &Consumer : Consumers)
    Consumer->HandleInterestingDecl(D);
}

void MultiplexConsumer::HandleTranslationUnit(ASTContext &Ctx) {
  for (auto &Consumer : Consumers)
    Consumer->HandleTranslationUnit(Ctx);
}

void MultiplexConsumer::HandleTagDeclDefinition(TagDecl *D) {
  for (auto &Consumer : Consumers)
    Consumer->HandleTagDeclDefinition(D);
}

void MultiplexConsumer::HandleTagDeclRequiredDefinition(const TagDecl *D) {
  for (auto &Consumer : Consumers)
    Consumer->HandleTagDeclRequiredDefinition(D);
}

void MultiplexConsumer::HandleCXXImplicitFunctionInstantiation(FunctionDecl *D){
  for (auto &Consumer : Consumers)
    Consumer->HandleCXXImplicitFunctionInstantiation(D);
}

void MultiplexConsumer::HandleTopLevelDeclInObjCContainer(DeclGroupRef D) {
  for (auto &Consumer : Consumers)
    Consumer->HandleTopLevelDeclInObjCContainer(D);
}

void MultiplexConsumer::HandleImplicitImportDecl(ImportDecl *D) {
  for (auto &Consumer : Consumers)
    Consumer->HandleImplicitImportDecl(D);
}

void MultiplexConsumer::HandleLinkerOptionPragma(llvm::StringRef Opts) {
  for (auto &Consumer : Consumers)
    Consumer->HandleLinkerOptionPragma(Opts);
}

void MultiplexConsumer::HandleDetectMismatch(llvm::StringRef Name, llvm::StringRef Value) {
  for (auto &Consumer : Consumers)
    Consumer->HandleDetectMismatch(Name, Value);
}

void MultiplexConsumer::HandleDependentLibrary(llvm::StringRef Lib) {
  for (auto &Consumer : Consumers)
    Consumer->HandleDependentLibrary(Lib);
}

void MultiplexConsumer::CompleteTentativeDefinition(VarDecl *D) {
  for (auto &Consumer : Consumers)
    Consumer->CompleteTentativeDefinition(D);
}

void MultiplexConsumer::HandleVTable(CXXRecordDecl *RD) {
  for (auto &Consumer : Consumers)
    Consumer->HandleVTable(RD);
}

ASTMutationListener *MultiplexConsumer::GetASTMutationListener() {
  return MutationListener.get();
}

ASTDeserializationListener *MultiplexConsumer::GetASTDeserializationListener() {
  return DeserializationListener.get();
}

void MultiplexConsumer::PrintStats() {
  for (auto &Consumer : Consumers)
    Consumer->PrintStats();
}

void MultiplexConsumer::InitializeSema(Sema &S) {
  for (auto &Consumer : Consumers)
    if (SemaConsumer *SC = dyn_cast<SemaConsumer>(Consumer.get()))
      SC->InitializeSema(S);
}

void MultiplexConsumer::ForgetSema() {
  for (auto &Consumer : Consumers)
    if (SemaConsumer *SC = dyn_cast<SemaConsumer>(Consumer.get()))
      SC->ForgetSema();
}
