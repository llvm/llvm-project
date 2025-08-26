//===- TemplateName.cpp - C++ Template Name Representation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TemplateName interface and subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/TemplateName.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DependenceFlags.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/OperatorKinds.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <optional>
#include <string>

using namespace clang;

DeducedTemplateStorage::DeducedTemplateStorage(TemplateName Underlying,
                                               const DefaultArguments &DefArgs)
    : UncommonTemplateNameStorage(Deduced, /*Index=*/DefArgs.StartPos,
                                  DefArgs.Args.size()),
      Underlying(Underlying) {
  llvm::copy(DefArgs.Args, reinterpret_cast<TemplateArgument *>(this + 1));
}

void DeducedTemplateStorage::Profile(llvm::FoldingSetNodeID &ID,
                                     const ASTContext &Context) const {
  Profile(ID, Context, Underlying, getDefaultArguments());
}

void DeducedTemplateStorage::Profile(llvm::FoldingSetNodeID &ID,
                                     const ASTContext &Context,
                                     TemplateName Underlying,
                                     const DefaultArguments &DefArgs) {
  Underlying.Profile(ID);
  ID.AddInteger(DefArgs.StartPos);
  ID.AddInteger(DefArgs.Args.size());
  for (const TemplateArgument &Arg : DefArgs.Args)
    Arg.Profile(ID, Context);
}

TemplateArgument
SubstTemplateTemplateParmPackStorage::getArgumentPack() const {
  return TemplateArgument(ArrayRef(Arguments, Bits.Data));
}

TemplateTemplateParmDecl *
SubstTemplateTemplateParmPackStorage::getParameterPack() const {
  return cast<TemplateTemplateParmDecl>(
      getReplacedTemplateParameterList(getAssociatedDecl())
          ->asArray()[Bits.Index]);
}

TemplateTemplateParmDecl *
SubstTemplateTemplateParmStorage::getParameter() const {
  return cast<TemplateTemplateParmDecl>(
      getReplacedTemplateParameterList(getAssociatedDecl())
          ->asArray()[Bits.Index]);
}

void SubstTemplateTemplateParmStorage::Profile(llvm::FoldingSetNodeID &ID) {
  Profile(ID, Replacement, getAssociatedDecl(), getIndex(), getPackIndex(),
          getFinal());
}

void SubstTemplateTemplateParmStorage::Profile(
    llvm::FoldingSetNodeID &ID, TemplateName Replacement, Decl *AssociatedDecl,
    unsigned Index, UnsignedOrNone PackIndex, bool Final) {
  Replacement.Profile(ID);
  ID.AddPointer(AssociatedDecl);
  ID.AddInteger(Index);
  ID.AddInteger(PackIndex.toInternalRepresentation());
  ID.AddBoolean(Final);
}

SubstTemplateTemplateParmPackStorage::SubstTemplateTemplateParmPackStorage(
    ArrayRef<TemplateArgument> ArgPack, Decl *AssociatedDecl, unsigned Index,
    bool Final)
    : UncommonTemplateNameStorage(SubstTemplateTemplateParmPack, Index,
                                  ArgPack.size()),
      Arguments(ArgPack.data()), AssociatedDeclAndFinal(AssociatedDecl, Final) {
  assert(AssociatedDecl != nullptr);
}

void SubstTemplateTemplateParmPackStorage::Profile(llvm::FoldingSetNodeID &ID,
                                                   ASTContext &Context) {
  Profile(ID, Context, getArgumentPack(), getAssociatedDecl(), getIndex(),
          getFinal());
}

Decl *SubstTemplateTemplateParmPackStorage::getAssociatedDecl() const {
  return AssociatedDeclAndFinal.getPointer();
}

bool SubstTemplateTemplateParmPackStorage::getFinal() const {
  return AssociatedDeclAndFinal.getInt();
}

void SubstTemplateTemplateParmPackStorage::Profile(
    llvm::FoldingSetNodeID &ID, ASTContext &Context,
    const TemplateArgument &ArgPack, Decl *AssociatedDecl, unsigned Index,
    bool Final) {
  ArgPack.Profile(ID, Context);
  ID.AddPointer(AssociatedDecl);
  ID.AddInteger(Index);
  ID.AddBoolean(Final);
}

IdentifierOrOverloadedOperator::IdentifierOrOverloadedOperator(
    const IdentifierInfo *II)
    : PtrOrOp(reinterpret_cast<uintptr_t>(II)) {
  static_assert(NUM_OVERLOADED_OPERATORS <= 4096,
                "NUM_OVERLOADED_OPERATORS is too large");
  assert(II);
  assert(getIdentifier() == II);
}
IdentifierOrOverloadedOperator::IdentifierOrOverloadedOperator(
    OverloadedOperatorKind OOK)
    : PtrOrOp(-uintptr_t(OOK)) {
  assert(OOK != OO_None);
  assert(getOperator() == OOK);
}

void IdentifierOrOverloadedOperator::Profile(llvm::FoldingSetNodeID &ID) const {
  if (auto *Identifier = getIdentifier()) {
    ID.AddBoolean(false);
    ID.AddPointer(Identifier);
  } else {
    ID.AddBoolean(true);
    ID.AddInteger(getOperator());
  }
}

TemplateName::TemplateName(void *Ptr) {
  Storage = StorageType::getFromOpaqueValue(Ptr);
}

TemplateName::TemplateName(TemplateDecl *Template) : Storage(Template) {}
TemplateName::TemplateName(OverloadedTemplateStorage *Storage)
    : Storage(Storage) {}
TemplateName::TemplateName(AssumedTemplateStorage *Storage)
    : Storage(Storage) {}
TemplateName::TemplateName(SubstTemplateTemplateParmStorage *Storage)
    : Storage(Storage) {}
TemplateName::TemplateName(SubstTemplateTemplateParmPackStorage *Storage)
    : Storage(Storage) {}
TemplateName::TemplateName(QualifiedTemplateName *Qual) : Storage(Qual) {}
TemplateName::TemplateName(DependentTemplateName *Dep) : Storage(Dep) {}
TemplateName::TemplateName(UsingShadowDecl *Using) : Storage(Using) {}
TemplateName::TemplateName(DeducedTemplateStorage *Deduced)
    : Storage(Deduced) {}

bool TemplateName::isNull() const { return Storage.isNull(); }

TemplateName::NameKind TemplateName::getKind() const {
  if (auto *ND = dyn_cast<Decl *>(Storage)) {
    if (isa<UsingShadowDecl>(ND))
      return UsingTemplate;
    assert(isa<TemplateDecl>(ND));
    return Template;
  }

  if (isa<DependentTemplateName *>(Storage))
    return DependentTemplate;
  if (isa<QualifiedTemplateName *>(Storage))
    return QualifiedTemplate;

  UncommonTemplateNameStorage *uncommon =
      cast<UncommonTemplateNameStorage *>(Storage);
  if (uncommon->getAsOverloadedStorage())
    return OverloadedTemplate;
  if (uncommon->getAsAssumedTemplateName())
    return AssumedTemplate;
  if (uncommon->getAsSubstTemplateTemplateParm())
    return SubstTemplateTemplateParm;
  if (uncommon->getAsDeducedTemplateName())
    return DeducedTemplate;

  assert(uncommon->getAsSubstTemplateTemplateParmPack() != nullptr);
  return SubstTemplateTemplateParmPack;
}

TemplateDecl *TemplateName::getAsTemplateDecl(bool IgnoreDeduced) const {
  TemplateName Name = *this;
  while (std::optional<TemplateName> UnderlyingOrNone =
             Name.desugar(IgnoreDeduced))
    Name = *UnderlyingOrNone;

  if (!IgnoreDeduced)
    assert(Name.getAsDeducedTemplateName() == nullptr &&
           "Unexpected canonical DeducedTemplateName; Did you mean to use "
           "getTemplateDeclAndDefaultArgs instead?");

  return cast_if_present<TemplateDecl>(
      dyn_cast_if_present<Decl *>(Name.Storage));
}

std::pair<TemplateDecl *, DefaultArguments>
TemplateName::getTemplateDeclAndDefaultArgs() const {
  for (TemplateName Name = *this; /**/; /**/) {
    if (Name.getKind() == TemplateName::DeducedTemplate) {
      DeducedTemplateStorage *DTS = Name.getAsDeducedTemplateName();
      TemplateDecl *TD =
          DTS->getUnderlying().getAsTemplateDecl(/*IgnoreDeduced=*/true);
      DefaultArguments DefArgs = DTS->getDefaultArguments();
      if (TD && DefArgs)
        assert(DefArgs.StartPos + DefArgs.Args.size() <=
               TD->getTemplateParameters()->size());
      return {TD, DTS->getDefaultArguments()};
    }
    if (std::optional<TemplateName> UnderlyingOrNone =
            Name.desugar(/*IgnoreDeduced=*/false)) {
      Name = *UnderlyingOrNone;
      continue;
    }
    return {cast_if_present<TemplateDecl>(Name.Storage.dyn_cast<Decl *>()), {}};
  }
}

std::optional<TemplateName> TemplateName::desugar(bool IgnoreDeduced) const {
  if (Decl *D = dyn_cast_if_present<Decl *>(Storage)) {
    if (auto *USD = dyn_cast<UsingShadowDecl>(D))
      return TemplateName(USD->getTargetDecl());
    return std::nullopt;
  }
  if (QualifiedTemplateName *QTN = getAsQualifiedTemplateName())
    return QTN->getUnderlyingTemplate();
  if (SubstTemplateTemplateParmStorage *S = getAsSubstTemplateTemplateParm())
    return S->getReplacement();
  if (IgnoreDeduced)
    if (DeducedTemplateStorage *S = getAsDeducedTemplateName())
      return S->getUnderlying();
  return std::nullopt;
}

OverloadedTemplateStorage *TemplateName::getAsOverloadedTemplate() const {
  if (UncommonTemplateNameStorage *Uncommon =
          Storage.dyn_cast<UncommonTemplateNameStorage *>())
    return Uncommon->getAsOverloadedStorage();

  return nullptr;
}

AssumedTemplateStorage *TemplateName::getAsAssumedTemplateName() const {
  if (UncommonTemplateNameStorage *Uncommon =
          Storage.dyn_cast<UncommonTemplateNameStorage *>())
    return Uncommon->getAsAssumedTemplateName();

  return nullptr;
}

SubstTemplateTemplateParmStorage *
TemplateName::getAsSubstTemplateTemplateParm() const {
  if (UncommonTemplateNameStorage *uncommon =
          dyn_cast_if_present<UncommonTemplateNameStorage *>(Storage))
    return uncommon->getAsSubstTemplateTemplateParm();

  return nullptr;
}

SubstTemplateTemplateParmPackStorage *
TemplateName::getAsSubstTemplateTemplateParmPack() const {
  if (UncommonTemplateNameStorage *Uncommon =
          Storage.dyn_cast<UncommonTemplateNameStorage *>())
    return Uncommon->getAsSubstTemplateTemplateParmPack();

  return nullptr;
}

QualifiedTemplateName *TemplateName::getAsQualifiedTemplateName() const {
  return dyn_cast_if_present<QualifiedTemplateName *>(Storage);
}

DependentTemplateName *TemplateName::getAsDependentTemplateName() const {
  return Storage.dyn_cast<DependentTemplateName *>();
}

std::tuple<NestedNameSpecifier, bool>
TemplateName::getQualifierAndTemplateKeyword() const {
  for (std::optional<TemplateName> Cur = *this; Cur;
       Cur = Cur->desugar(/*IgnoreDeduced=*/true)) {
    if (DependentTemplateName *N = Cur->getAsDependentTemplateName())
      return {N->getQualifier(), N->hasTemplateKeyword()};
    if (QualifiedTemplateName *N = Cur->getAsQualifiedTemplateName())
      return {N->getQualifier(), N->hasTemplateKeyword()};
    if (Cur->getAsSubstTemplateTemplateParm() ||
        Cur->getAsSubstTemplateTemplateParmPack())
      break;
  }
  return {std::nullopt, false};
}

UsingShadowDecl *TemplateName::getAsUsingShadowDecl() const {
  if (Decl *D = Storage.dyn_cast<Decl *>())
    if (UsingShadowDecl *USD = dyn_cast<UsingShadowDecl>(D))
      return USD;
  if (QualifiedTemplateName *QTN = getAsQualifiedTemplateName())
    return QTN->getUnderlyingTemplate().getAsUsingShadowDecl();
  return nullptr;
}

DependentTemplateStorage::DependentTemplateStorage(
    NestedNameSpecifier Qualifier, IdentifierOrOverloadedOperator Name,
    bool HasTemplateKeyword)
    : Qualifier(Qualifier, HasTemplateKeyword), Name(Name) {
  assert((!Qualifier || Qualifier.isDependent()) &&
         "Qualifier must be dependent");
}

TemplateNameDependence DependentTemplateStorage::getDependence() const {
  return toTemplateNameDependence(getQualifier().getDependence()) |
         TemplateNameDependence::DependentInstantiation;
}

void DependentTemplateStorage::print(raw_ostream &OS,
                                     const PrintingPolicy &Policy) const {
  getQualifier().print(OS, Policy);

  if (hasTemplateKeyword())
    OS << "template ";

  IdentifierOrOverloadedOperator Name = getName();
  if (const IdentifierInfo *II = Name.getIdentifier())
    OS << II->getName();
  else
    OS << "operator " << getOperatorSpelling(Name.getOperator());
}

DeducedTemplateStorage *TemplateName::getAsDeducedTemplateName() const {
  if (UncommonTemplateNameStorage *Uncommon =
          dyn_cast_if_present<UncommonTemplateNameStorage *>(Storage))
    return Uncommon->getAsDeducedTemplateName();

  return nullptr;
}

TemplateNameDependence TemplateName::getDependence() const {
  switch (getKind()) {
  case NameKind::Template:
  case NameKind::UsingTemplate: {
    TemplateDecl *Template = getAsTemplateDecl();
    auto D = TemplateNameDependence::None;
    if (auto *TTP = dyn_cast<TemplateTemplateParmDecl>(Template)) {
      D |= TemplateNameDependence::DependentInstantiation;
      if (TTP->isParameterPack())
        D |= TemplateNameDependence::UnexpandedPack;
    }
    // FIXME: Hack, getDeclContext() can be null if Template is still
    // initializing due to PCH reading, so we check it before using it.
    // Should probably modify TemplateSpecializationType to allow constructing
    // it without the isDependent() checking.
    if (Template->getDeclContext() &&
        Template->getDeclContext()->isDependentContext())
      D |= TemplateNameDependence::DependentInstantiation;
    return D;
  }
  case NameKind::QualifiedTemplate: {
    QualifiedTemplateName *S = getAsQualifiedTemplateName();
    TemplateNameDependence D = S->getUnderlyingTemplate().getDependence();
    D |= toTemplateNameDependence(S->getQualifier().getDependence());
    return D;
  }
  case NameKind::DependentTemplate: {
    DependentTemplateName *S = getAsDependentTemplateName();
    return toTemplateNameDependence(S->getQualifier().getDependence()) |
           TemplateNameDependence::DependentInstantiation;
  }
  case NameKind::SubstTemplateTemplateParm: {
    auto *S = getAsSubstTemplateTemplateParm();
    return S->getReplacement().getDependence();
  }
  case NameKind::SubstTemplateTemplateParmPack:
    return TemplateNameDependence::UnexpandedPack |
           TemplateNameDependence::DependentInstantiation;
  case NameKind::DeducedTemplate: {
    DeducedTemplateStorage *DTS = getAsDeducedTemplateName();
    TemplateNameDependence D = DTS->getUnderlying().getDependence();
    for (const TemplateArgument &Arg : DTS->getDefaultArguments().Args)
      D |= toTemplateNameDependence(Arg.getDependence());
    return D;
  }
  case NameKind::AssumedTemplate:
    return TemplateNameDependence::DependentInstantiation;
  case NameKind::OverloadedTemplate:
    llvm_unreachable("overloaded templates shouldn't survive to here.");
  }
  llvm_unreachable("Unknown TemplateName kind");
}

bool TemplateName::isDependent() const {
  return getDependence() & TemplateNameDependence::Dependent;
}

bool TemplateName::isInstantiationDependent() const {
  return getDependence() & TemplateNameDependence::Instantiation;
}

bool TemplateName::containsUnexpandedParameterPack() const {
  return getDependence() & TemplateNameDependence::UnexpandedPack;
}

void TemplateName::print(raw_ostream &OS, const PrintingPolicy &Policy,
                         Qualified Qual) const {
  auto handleAnonymousTTP = [&](TemplateDecl *TD, raw_ostream &OS) {
    if (TemplateTemplateParmDecl *TTP = dyn_cast<TemplateTemplateParmDecl>(TD);
        TTP && (Policy.PrintAsCanonical || TTP->getIdentifier() == nullptr)) {
      OS << "template-parameter-" << TTP->getDepth() << "-" << TTP->getIndex();
      return true;
    }
    return false;
  };
  if (NameKind Kind = getKind();
      Kind == TemplateName::Template || Kind == TemplateName::UsingTemplate) {
    // After `namespace ns { using std::vector }`, what is the fully-qualified
    // name of the UsingTemplateName `vector` within ns?
    //
    // - ns::vector (the qualified name of the using-shadow decl)
    // - std::vector (the qualified name of the underlying template decl)
    //
    // Similar to the UsingType behavior, using declarations are used to import
    // names more often than to export them, thus using the original name is
    // most useful in this case.
    TemplateDecl *Template = getAsTemplateDecl();
    if (Policy.PrintAsCanonical)
      Template = cast<TemplateDecl>(Template->getCanonicalDecl());
    if (handleAnonymousTTP(Template, OS))
      return;
    if (Qual == Qualified::None || isa<TemplateTemplateParmDecl>(Template) ||
        Policy.SuppressScope) {
      if (IdentifierInfo *II = Template->getIdentifier();
          Policy.CleanUglifiedParameters && II &&
          isa<TemplateTemplateParmDecl>(Template))
        OS << II->deuglifiedName();
      else
        OS << *Template;
    } else {
      PrintingPolicy NestedNamePolicy = Policy;
      NestedNamePolicy.SuppressUnwrittenScope = true;
      Template->printQualifiedName(OS, NestedNamePolicy);
    }
  } else if (QualifiedTemplateName *QTN = getAsQualifiedTemplateName()) {
    if (Policy.PrintAsCanonical) {
      QTN->getUnderlyingTemplate().print(OS, Policy, Qual);
      return;
    }
    if (Qual != Qualified::None)
      QTN->getQualifier().print(OS, Policy);
    if (QTN->hasTemplateKeyword())
      OS << "template ";

    TemplateName Underlying = QTN->getUnderlyingTemplate();
    assert(Underlying.getKind() == TemplateName::Template ||
           Underlying.getKind() == TemplateName::UsingTemplate);

    TemplateDecl *UTD = Underlying.getAsTemplateDecl();

    if (handleAnonymousTTP(UTD, OS))
      return;

    OS << *UTD;
  } else if (DependentTemplateName *DTN = getAsDependentTemplateName()) {
    DTN->print(OS, Policy);
  } else if (SubstTemplateTemplateParmStorage *subst =
                 getAsSubstTemplateTemplateParm()) {
    subst->getReplacement().print(OS, Policy, Qual);
  } else if (SubstTemplateTemplateParmPackStorage *SubstPack =
                 getAsSubstTemplateTemplateParmPack())
    OS << *SubstPack->getParameterPack();
  else if (AssumedTemplateStorage *Assumed = getAsAssumedTemplateName()) {
    Assumed->getDeclName().print(OS, Policy);
  } else if (DeducedTemplateStorage *Deduced = getAsDeducedTemplateName()) {
    Deduced->getUnderlying().print(OS, Policy);
    DefaultArguments DefArgs = Deduced->getDefaultArguments();
    OS << ":" << DefArgs.StartPos;
    printTemplateArgumentList(OS, DefArgs.Args, Policy);
  } else {
    assert(getKind() == TemplateName::OverloadedTemplate);
    OverloadedTemplateStorage *OTS = getAsOverloadedTemplate();
    (*OTS->begin())->printName(OS, Policy);
  }
}

const StreamingDiagnostic &clang::operator<<(const StreamingDiagnostic &DB,
                                             TemplateName N) {
  std::string NameStr;
  llvm::raw_string_ostream OS(NameStr);
  LangOptions LO;
  LO.CPlusPlus = true;
  LO.Bool = true;
  OS << '\'';
  N.print(OS, PrintingPolicy(LO));
  OS << '\'';
  return DB << NameStr;
}
