//===--- TemplateBase.cpp - Common template AST class implementation ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements common classes used throughout C++ template
// representations.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/TemplateBase.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallString.h"
#include <algorithm>
#include <cctype>

using namespace clang;

/// \brief Print a template integral argument value.
///
/// \param TemplArg the TemplateArgument instance to print.
///
/// \param Out the raw_ostream instance to use for printing.
static void printIntegral(const TemplateArgument &TemplArg,
                          raw_ostream &Out) {
  const ::clang::Type *T = TemplArg.getIntegralType().getTypePtr();
  const llvm::APSInt &Val = TemplArg.getAsIntegral();

  if (T->isBooleanType()) {
    Out << (Val.getBoolValue() ? "true" : "false");
  } else if (T->isCharType()) {
    const char Ch = Val.getZExtValue();
    Out << ((Ch == '\'') ? "'\\" : "'");
    Out.write_escaped(StringRef(&Ch, 1), /*UseHexEscapes=*/ true);
    Out << "'";
  } else {
    Out << Val;
  }
}

//===----------------------------------------------------------------------===//
// TemplateArgument Implementation
//===----------------------------------------------------------------------===//

TemplateArgument::TemplateArgument(ASTContext &Ctx, const llvm::APSInt &Value,
                                   QualType Type)
  : Kind(Integral) {
  // Copy the APSInt value into our decomposed form.
  Integer.BitWidth = Value.getBitWidth();
  Integer.IsUnsigned = Value.isUnsigned();
  // If the value is large, we have to get additional memory from the ASTContext
  unsigned NumWords = Value.getNumWords();
  if (NumWords > 1) {
    void *Mem = Ctx.Allocate(NumWords * sizeof(uint64_t));
    std::memcpy(Mem, Value.getRawData(), NumWords * sizeof(uint64_t));
    Integer.pVal = static_cast<uint64_t *>(Mem);
  } else {
    Integer.VAL = Value.getZExtValue();
  }

  Integer.Type = Type.getAsOpaquePtr();
}

TemplateArgument TemplateArgument::CreatePackCopy(ASTContext &Context,
                                                  const TemplateArgument *Args,
                                                  unsigned NumArgs) {
  if (NumArgs == 0)
    return getEmptyPack();
  
  TemplateArgument *Storage = new (Context) TemplateArgument [NumArgs];
  std::copy(Args, Args + NumArgs, Storage);
  return TemplateArgument(Storage, NumArgs);
}

bool TemplateArgument::isDependent() const {
  switch (getKind()) {
  case Null:
    llvm_unreachable("Should not have a NULL template argument");

  case Type:
    return getAsType()->isDependentType();

  case Template:
    return getAsTemplate().isDependent();

  case TemplateExpansion:
    return true;

  case Declaration:
    if (DeclContext *DC = dyn_cast<DeclContext>(getAsDecl()))
      return DC->isDependentContext();
    return getAsDecl()->getDeclContext()->isDependentContext();

  case NullPtr:
    return false;

  case Integral:
    // Never dependent
    return false;

  case Expression:
    return (getAsExpr()->isTypeDependent() || getAsExpr()->isValueDependent());

  case Pack:
    for (pack_iterator P = pack_begin(), PEnd = pack_end(); P != PEnd; ++P) {
      if (P->isDependent())
        return true;
    }

    return false;
  }

  llvm_unreachable("Invalid TemplateArgument Kind!");
}

bool TemplateArgument::isInstantiationDependent() const {
  switch (getKind()) {
  case Null:
    llvm_unreachable("Should not have a NULL template argument");
    
  case Type:
    return getAsType()->isInstantiationDependentType();
    
  case Template:
    return getAsTemplate().isInstantiationDependent();
    
  case TemplateExpansion:
    return true;
    
  case Declaration:
    if (DeclContext *DC = dyn_cast<DeclContext>(getAsDecl()))
      return DC->isDependentContext();
    return getAsDecl()->getDeclContext()->isDependentContext();

  case NullPtr:
    return false;
      
  case Integral:
    // Never dependent
    return false;
    
  case Expression:
    return getAsExpr()->isInstantiationDependent();
    
  case Pack:
    for (pack_iterator P = pack_begin(), PEnd = pack_end(); P != PEnd; ++P) {
      if (P->isInstantiationDependent())
        return true;
    }
    
    return false;
  }

  llvm_unreachable("Invalid TemplateArgument Kind!");
}

bool TemplateArgument::isPackExpansion() const {
  switch (getKind()) {
  case Null:
  case Declaration:
  case Integral:
  case Pack:    
  case Template:
  case NullPtr:
    return false;
      
  case TemplateExpansion:
    return true;
      
  case Type:
    return isa<PackExpansionType>(getAsType());
          
  case Expression:
    return isa<PackExpansionExpr>(getAsExpr());
  }

  llvm_unreachable("Invalid TemplateArgument Kind!");
}

bool TemplateArgument::containsUnexpandedParameterPack() const {
  switch (getKind()) {
  case Null:
  case Declaration:
  case Integral:
  case TemplateExpansion:
  case NullPtr:
    break;

  case Type:
    if (getAsType()->containsUnexpandedParameterPack())
      return true;
    break;

  case Template:
    if (getAsTemplate().containsUnexpandedParameterPack())
      return true;
    break;
        
  case Expression:
    if (getAsExpr()->containsUnexpandedParameterPack())
      return true;
    break;

  case Pack:
    for (pack_iterator P = pack_begin(), PEnd = pack_end(); P != PEnd; ++P)
      if (P->containsUnexpandedParameterPack())
        return true;

    break;
  }

  return false;
}

llvm::Optional<unsigned> TemplateArgument::getNumTemplateExpansions() const {
  assert(Kind == TemplateExpansion);
  if (TemplateArg.NumExpansions)
    return TemplateArg.NumExpansions - 1;
  
  return llvm::Optional<unsigned>();
}

void TemplateArgument::Profile(llvm::FoldingSetNodeID &ID,
                               const ASTContext &Context) const {
  ID.AddInteger(Kind);
  switch (Kind) {
  case Null:
    break;

  case Type:
    getAsType().Profile(ID);
    break;

  case Declaration:
    ID.AddPointer(getAsDecl()? getAsDecl()->getCanonicalDecl() : 0);
    break;

  case Template:
  case TemplateExpansion: {
    TemplateName Template = getAsTemplateOrTemplatePattern();
    if (TemplateTemplateParmDecl *TTP
          = dyn_cast_or_null<TemplateTemplateParmDecl>(
                                                Template.getAsTemplateDecl())) {
      ID.AddBoolean(true);
      ID.AddInteger(TTP->getDepth());
      ID.AddInteger(TTP->getPosition());
      ID.AddBoolean(TTP->isParameterPack());
    } else {
      ID.AddBoolean(false);
      ID.AddPointer(Context.getCanonicalTemplateName(Template)
                                                          .getAsVoidPointer());
    }
    break;
  }
      
  case Integral:
    getAsIntegral().Profile(ID);
    getIntegralType().Profile(ID);
    break;

  case Expression:
    getAsExpr()->Profile(ID, Context, true);
    break;

  case Pack:
    ID.AddInteger(Args.NumArgs);
    for (unsigned I = 0; I != Args.NumArgs; ++I)
      Args.Args[I].Profile(ID, Context);
  }
}

bool TemplateArgument::structurallyEquals(const TemplateArgument &Other) const {
  if (getKind() != Other.getKind()) return false;

  switch (getKind()) {
  case Null:
  case Type:
  case Expression:      
  case Template:
  case TemplateExpansion:
  case NullPtr:
    return TypeOrValue == Other.TypeOrValue;

  case Declaration:
    return getAsDecl() == Other.getAsDecl() && 
           isDeclForReferenceParam() && Other.isDeclForReferenceParam();

  case Integral:
    return getIntegralType() == Other.getIntegralType() &&
           getAsIntegral() == Other.getAsIntegral();

  case Pack:
    if (Args.NumArgs != Other.Args.NumArgs) return false;
    for (unsigned I = 0, E = Args.NumArgs; I != E; ++I)
      if (!Args.Args[I].structurallyEquals(Other.Args.Args[I]))
        return false;
    return true;
  }

  llvm_unreachable("Invalid TemplateArgument Kind!");
}

TemplateArgument TemplateArgument::getPackExpansionPattern() const {
  assert(isPackExpansion());
  
  switch (getKind()) {
  case Type:
    return getAsType()->getAs<PackExpansionType>()->getPattern();
    
  case Expression:
    return cast<PackExpansionExpr>(getAsExpr())->getPattern();
    
  case TemplateExpansion:
    return TemplateArgument(getAsTemplateOrTemplatePattern());

  case Declaration:
  case Integral:
  case Pack:
  case Null:
  case Template:
  case NullPtr:
    return TemplateArgument();
  }

  llvm_unreachable("Invalid TemplateArgument Kind!");
}

void TemplateArgument::print(const PrintingPolicy &Policy, 
                             raw_ostream &Out) const {
  switch (getKind()) {
  case Null:
    Out << "<no value>";
    break;
    
  case Type: {
    PrintingPolicy SubPolicy(Policy);
    SubPolicy.SuppressStrongLifetime = true;
    std::string TypeStr;
    getAsType().getAsStringInternal(TypeStr, SubPolicy);
    Out << TypeStr;
    break;
  }
    
  case Declaration: {
    NamedDecl *ND = cast<NamedDecl>(getAsDecl());
    if (ND->getDeclName()) {
      // FIXME: distinguish between pointer and reference args?
      Out << *ND;
    } else {
      Out << "<anonymous>";
    }
    break;
  }

  case NullPtr:
    Out << "nullptr";
    break;

  case Template:
    getAsTemplate().print(Out, Policy);
    break;

  case TemplateExpansion:
    getAsTemplateOrTemplatePattern().print(Out, Policy);
    Out << "...";
    break;
      
  case Integral: {
    printIntegral(*this, Out);
    break;
  }
    
  case Expression:
    getAsExpr()->printPretty(Out, 0, Policy);
    break;
    
  case Pack:
    Out << "<";
    bool First = true;
    for (TemplateArgument::pack_iterator P = pack_begin(), PEnd = pack_end();
         P != PEnd; ++P) {
      if (First)
        First = false;
      else
        Out << ", ";
      
      P->print(Policy, Out);
    }
    Out << ">";
    break;        
  }
}

//===----------------------------------------------------------------------===//
// TemplateArgumentLoc Implementation
//===----------------------------------------------------------------------===//

TemplateArgumentLocInfo::TemplateArgumentLocInfo() {
  memset((void*)this, 0, sizeof(TemplateArgumentLocInfo));
}

SourceRange TemplateArgumentLoc::getSourceRange() const {
  switch (Argument.getKind()) {
  case TemplateArgument::Expression:
    return getSourceExpression()->getSourceRange();

  case TemplateArgument::Declaration:
    return getSourceDeclExpression()->getSourceRange();

  case TemplateArgument::NullPtr:
    return getSourceNullPtrExpression()->getSourceRange();

  case TemplateArgument::Type:
    if (TypeSourceInfo *TSI = getTypeSourceInfo())
      return TSI->getTypeLoc().getSourceRange();
    else
      return SourceRange();

  case TemplateArgument::Template:
    if (getTemplateQualifierLoc())
      return SourceRange(getTemplateQualifierLoc().getBeginLoc(), 
                         getTemplateNameLoc());
    return SourceRange(getTemplateNameLoc());

  case TemplateArgument::TemplateExpansion:
    if (getTemplateQualifierLoc())
      return SourceRange(getTemplateQualifierLoc().getBeginLoc(), 
                         getTemplateEllipsisLoc());
    return SourceRange(getTemplateNameLoc(), getTemplateEllipsisLoc());

  case TemplateArgument::Integral:
    return getSourceIntegralExpression()->getSourceRange();

  case TemplateArgument::Pack:
  case TemplateArgument::Null:
    return SourceRange();
  }

  llvm_unreachable("Invalid TemplateArgument Kind!");
}

TemplateArgumentLoc 
TemplateArgumentLoc::getPackExpansionPattern(SourceLocation &Ellipsis,
                                       llvm::Optional<unsigned> &NumExpansions,
                                             ASTContext &Context) const {
  assert(Argument.isPackExpansion());
  
  switch (Argument.getKind()) {
  case TemplateArgument::Type: {
    // FIXME: We shouldn't ever have to worry about missing
    // type-source info!
    TypeSourceInfo *ExpansionTSInfo = getTypeSourceInfo();
    if (!ExpansionTSInfo)
      ExpansionTSInfo = Context.getTrivialTypeSourceInfo(
                                                     getArgument().getAsType(),
                                                         Ellipsis);
    PackExpansionTypeLoc Expansion
      = cast<PackExpansionTypeLoc>(ExpansionTSInfo->getTypeLoc());
    Ellipsis = Expansion.getEllipsisLoc();
    
    TypeLoc Pattern = Expansion.getPatternLoc();
    NumExpansions = Expansion.getTypePtr()->getNumExpansions();
    
    // FIXME: This is horrible. We know where the source location data is for
    // the pattern, and we have the pattern's type, but we are forced to copy
    // them into an ASTContext because TypeSourceInfo bundles them together
    // and TemplateArgumentLoc traffics in TypeSourceInfo pointers.
    TypeSourceInfo *PatternTSInfo
      = Context.CreateTypeSourceInfo(Pattern.getType(),
                                     Pattern.getFullDataSize());
    memcpy(PatternTSInfo->getTypeLoc().getOpaqueData(), 
           Pattern.getOpaqueData(), Pattern.getFullDataSize());
    return TemplateArgumentLoc(TemplateArgument(Pattern.getType()),
                               PatternTSInfo);
  }
      
  case TemplateArgument::Expression: {
    PackExpansionExpr *Expansion
      = cast<PackExpansionExpr>(Argument.getAsExpr());
    Expr *Pattern = Expansion->getPattern();
    Ellipsis = Expansion->getEllipsisLoc();
    NumExpansions = Expansion->getNumExpansions();
    return TemplateArgumentLoc(Pattern, Pattern);
  }

  case TemplateArgument::TemplateExpansion:
    Ellipsis = getTemplateEllipsisLoc();
    NumExpansions = Argument.getNumTemplateExpansions();
    return TemplateArgumentLoc(Argument.getPackExpansionPattern(),
                               getTemplateQualifierLoc(),
                               getTemplateNameLoc());
    
  case TemplateArgument::Declaration:
  case TemplateArgument::NullPtr:
  case TemplateArgument::Template:
  case TemplateArgument::Integral:
  case TemplateArgument::Pack:
  case TemplateArgument::Null:
    return TemplateArgumentLoc();
  }

  llvm_unreachable("Invalid TemplateArgument Kind!");
}

const DiagnosticBuilder &clang::operator<<(const DiagnosticBuilder &DB,
                                           const TemplateArgument &Arg) {
  switch (Arg.getKind()) {
  case TemplateArgument::Null:
    // This is bad, but not as bad as crashing because of argument
    // count mismatches.
    return DB << "(null template argument)";
      
  case TemplateArgument::Type:
    return DB << Arg.getAsType();
      
  case TemplateArgument::Declaration:
    return DB << Arg.getAsDecl();

  case TemplateArgument::NullPtr:
    return DB << "nullptr";
      
  case TemplateArgument::Integral:
    return DB << Arg.getAsIntegral().toString(10);
      
  case TemplateArgument::Template:
    return DB << Arg.getAsTemplate();

  case TemplateArgument::TemplateExpansion:
    return DB << Arg.getAsTemplateOrTemplatePattern() << "...";

  case TemplateArgument::Expression: {
    // This shouldn't actually ever happen, so it's okay that we're
    // regurgitating an expression here.
    // FIXME: We're guessing at LangOptions!
    SmallString<32> Str;
    llvm::raw_svector_ostream OS(Str);
    LangOptions LangOpts;
    LangOpts.CPlusPlus = true;
    PrintingPolicy Policy(LangOpts);
    Arg.getAsExpr()->printPretty(OS, 0, Policy);
    return DB << OS.str();
  }
      
  case TemplateArgument::Pack: {
    // FIXME: We're guessing at LangOptions!
    SmallString<32> Str;
    llvm::raw_svector_ostream OS(Str);
    LangOptions LangOpts;
    LangOpts.CPlusPlus = true;
    PrintingPolicy Policy(LangOpts);
    Arg.print(Policy, OS);
    return DB << OS.str();
  }
  }

  llvm_unreachable("Invalid TemplateArgument Kind!");
}

const ASTTemplateArgumentListInfo *
ASTTemplateArgumentListInfo::Create(ASTContext &C,
                                    const TemplateArgumentListInfo &List) {
  std::size_t size = ASTTemplateArgumentListInfo::sizeFor(List.size());
  void *Mem = C.Allocate(size, llvm::alignOf<ASTTemplateArgumentListInfo>());
  ASTTemplateArgumentListInfo *TAI = new (Mem) ASTTemplateArgumentListInfo();
  TAI->initializeFrom(List);
  return TAI;
}

void ASTTemplateArgumentListInfo::initializeFrom(
                                      const TemplateArgumentListInfo &Info) {
  LAngleLoc = Info.getLAngleLoc();
  RAngleLoc = Info.getRAngleLoc();
  NumTemplateArgs = Info.size();

  TemplateArgumentLoc *ArgBuffer = getTemplateArgs();
  for (unsigned i = 0; i != NumTemplateArgs; ++i)
    new (&ArgBuffer[i]) TemplateArgumentLoc(Info[i]);
}

void ASTTemplateArgumentListInfo::initializeFrom(
                                          const TemplateArgumentListInfo &Info,
                                                  bool &Dependent, 
                                                  bool &InstantiationDependent,
                                       bool &ContainsUnexpandedParameterPack) {
  LAngleLoc = Info.getLAngleLoc();
  RAngleLoc = Info.getRAngleLoc();
  NumTemplateArgs = Info.size();

  TemplateArgumentLoc *ArgBuffer = getTemplateArgs();
  for (unsigned i = 0; i != NumTemplateArgs; ++i) {
    Dependent = Dependent || Info[i].getArgument().isDependent();
    InstantiationDependent = InstantiationDependent || 
                             Info[i].getArgument().isInstantiationDependent();
    ContainsUnexpandedParameterPack 
      = ContainsUnexpandedParameterPack || 
        Info[i].getArgument().containsUnexpandedParameterPack();

    new (&ArgBuffer[i]) TemplateArgumentLoc(Info[i]);
  }
}

void ASTTemplateArgumentListInfo::copyInto(
                                      TemplateArgumentListInfo &Info) const {
  Info.setLAngleLoc(LAngleLoc);
  Info.setRAngleLoc(RAngleLoc);
  for (unsigned I = 0; I != NumTemplateArgs; ++I)
    Info.addArgument(getTemplateArgs()[I]);
}

std::size_t ASTTemplateArgumentListInfo::sizeFor(unsigned NumTemplateArgs) {
  return sizeof(ASTTemplateArgumentListInfo) +
         sizeof(TemplateArgumentLoc) * NumTemplateArgs;
}

void
ASTTemplateKWAndArgsInfo::initializeFrom(SourceLocation TemplateKWLoc,
                                         const TemplateArgumentListInfo &Info) {
  Base::initializeFrom(Info);
  setTemplateKeywordLoc(TemplateKWLoc);
}

void
ASTTemplateKWAndArgsInfo
::initializeFrom(SourceLocation TemplateKWLoc,
                 const TemplateArgumentListInfo &Info,
                 bool &Dependent,
                 bool &InstantiationDependent,
                 bool &ContainsUnexpandedParameterPack) {
  Base::initializeFrom(Info, Dependent, InstantiationDependent,
                       ContainsUnexpandedParameterPack);
  setTemplateKeywordLoc(TemplateKWLoc);
}

void
ASTTemplateKWAndArgsInfo::initializeFrom(SourceLocation TemplateKWLoc) {
  // No explicit template arguments, but template keyword loc is valid.
  assert(TemplateKWLoc.isValid());
  LAngleLoc = SourceLocation();
  RAngleLoc = SourceLocation();
  NumTemplateArgs = 0;
  setTemplateKeywordLoc(TemplateKWLoc);
}

std::size_t
ASTTemplateKWAndArgsInfo::sizeFor(unsigned NumTemplateArgs) {
  // Add space for the template keyword location.
  // FIXME: There's room for this in the padding before the template args in
  //        64-bit builds.
  return Base::sizeFor(NumTemplateArgs) + sizeof(SourceLocation);
}
