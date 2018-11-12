//===--- IndexRecordHasher.cpp - Index record hashing ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IndexRecordHasher.h"
#include "FileIndexRecord.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclVisitor.h"
#include "llvm/Support/Path.h"

#define INITIAL_HASH 5381
#define COMBINE_HASH(...) (Hash = hash_combine(Hash, __VA_ARGS__))

using namespace clang;
using namespace clang::index;
using namespace llvm;

static hash_code computeHash(const TemplateArgument &Arg,
                             IndexRecordHasher &Hasher);

namespace {
class DeclHashVisitor : public ConstDeclVisitor<DeclHashVisitor, hash_code> {
  IndexRecordHasher &Hasher;

public:
  DeclHashVisitor(IndexRecordHasher &Hasher) : Hasher(Hasher) {}

  hash_code VisitDecl(const Decl *D) {
    return VisitDeclContext(D->getDeclContext());
  }

  hash_code VisitNamedDecl(const NamedDecl *D) {
    hash_code Hash = VisitDecl(D);
    if (auto *attr = D->getExternalSourceSymbolAttr()) {
      COMBINE_HASH(hash_value(attr->getDefinedIn()));
    }
    return COMBINE_HASH(Hasher.hash(D->getDeclName()));
  }

  hash_code VisitTagDecl(const TagDecl *D) {
    if (D->getDeclName().isEmpty()) {
      if (const TypedefNameDecl *TD = D->getTypedefNameForAnonDecl())
        return Visit(TD);

      hash_code Hash = VisitDeclContext(D->getDeclContext());
      if (D->isEmbeddedInDeclarator() && !D->isFreeStanding()) {
        COMBINE_HASH(hashLoc(D->getLocation(), /*IncludeOffset=*/true));
      } else
        COMBINE_HASH('a');
      return Hash;
    }

    hash_code Hash = VisitTypeDecl(D);
    return COMBINE_HASH('T');
  }

  hash_code VisitClassTemplateSpecializationDecl(const ClassTemplateSpecializationDecl *D) {
    hash_code Hash = VisitCXXRecordDecl(D);
    const TemplateArgumentList &Args = D->getTemplateArgs();
    COMBINE_HASH('>');
    for (unsigned I = 0, N = Args.size(); I != N; ++I) {
      COMBINE_HASH(computeHash(Args.get(I), Hasher));
    }
    return Hash;
  }

  hash_code VisitObjCContainerDecl(const ObjCContainerDecl *D) {
    hash_code Hash = VisitNamedDecl(D);
    return COMBINE_HASH('I');
  }

  hash_code VisitObjCImplDecl(const ObjCImplDecl *D) {
    if (auto *ID = D->getClassInterface())
      return VisitObjCInterfaceDecl(ID);
    else
      return 0;
  }

  hash_code VisitObjCCategoryDecl(const ObjCCategoryDecl *D) {
    // FIXME: Differentiate between category and the interface ?
    if (auto *ID = D->getClassInterface())
      return VisitObjCInterfaceDecl(ID);
    else
      return 0;
  }

  hash_code VisitFunctionDecl(const FunctionDecl *D) {
    hash_code Hash = VisitNamedDecl(D);
    ASTContext &Ctx = Hasher.getASTContext();
    if ((!Ctx.getLangOpts().CPlusPlus && !D->hasAttr<OverloadableAttr>())
        || D->isExternC())
      return Hash;

    for (auto param : D->parameters()) {
      COMBINE_HASH(Hasher.hash(param->getType()));
    }
    return Hash;
  }

  hash_code VisitUnresolvedUsingTypenameDecl(const UnresolvedUsingTypenameDecl *D) {
    hash_code Hash = VisitNamedDecl(D);
    COMBINE_HASH(Hasher.hash(D->getQualifier()));
    return Hash;
  }

  hash_code VisitUnresolvedUsingValueDecl(const UnresolvedUsingValueDecl *D) {
    hash_code Hash = VisitNamedDecl(D);
    COMBINE_HASH(Hasher.hash(D->getQualifier()));
    return Hash;
  }

  hash_code VisitDeclContext(const DeclContext *DC) {
    // FIXME: Add location if this is anonymous namespace ?
    DC = DC->getRedeclContext();
    const Decl *D = cast<Decl>(DC)->getCanonicalDecl();
    if (auto *ND = dyn_cast<NamedDecl>(D))
      return Hasher.hash(ND);
    else
      return 0;
  }

  hash_code hashLoc(SourceLocation Loc, bool IncludeOffset) {
    if (Loc.isInvalid()) {
      return 0;
    }
    hash_code Hash = INITIAL_HASH;
    const SourceManager &SM = Hasher.getASTContext().getSourceManager();
    Loc = SM.getFileLoc(Loc);
    const std::pair<FileID, unsigned> &Decomposed = SM.getDecomposedLoc(Loc);
    const FileEntry *FE = SM.getFileEntryForID(Decomposed.first);
    if (FE) {
      COMBINE_HASH(llvm::sys::path::filename(FE->getName()));
    } else {
      // This case really isn't interesting.
      return 0;
    }
    if (IncludeOffset) {
      // Use the offest into the FileID to represent the location.  Using
      // a line/column can cause us to look back at the original source file,
      // which is expensive.
      COMBINE_HASH(Decomposed.second);
    }
    return Hash;
  }
};
}

hash_code IndexRecordHasher::hashRecord(const FileIndexRecord &Record) {
  hash_code Hash = INITIAL_HASH;
  for (auto &Info : Record.getDeclOccurrences()) {
    COMBINE_HASH(Info.Roles, Info.Offset, hash(Info.Dcl));
    for (auto &Rel : Info.Relations) {
      COMBINE_HASH(hash(Rel.RelatedSymbol));
    }
  }
  return Hash;
}

hash_code IndexRecordHasher::hash(const Decl *D) {
  assert(D->isCanonicalDecl());

  if (isa<TagDecl>(D) || isa<ObjCContainerDecl>(D)) {
    return tryCache(D, D);
  } else  if (auto *NS = dyn_cast<NamespaceDecl>(D)) {
    if (NS->isAnonymousNamespace())
      return hash_value(StringRef("@aN"));
    return tryCache(D, D);
  } else {
    // There's a balance between caching results and not growing the cache too
    // much. Measurements showed that avoiding caching all decls is beneficial
    // particularly when including all of Cocoa.
    return hashImpl(D);
  }
}

hash_code IndexRecordHasher::hash(QualType NonCanTy) {
  CanQualType CanTy = Ctx.getCanonicalType(NonCanTy);
  return hash(CanTy);
}

hash_code IndexRecordHasher::hash(CanQualType CT) {
  // Do some hashing without going to the cache, for example we can avoid
  // storing the hash for both the type and its const-qualified version.
  hash_code Hash = INITIAL_HASH;

  auto asCanon = [](QualType Ty) -> CanQualType {
    return CanQualType::CreateUnsafe(Ty);
  };

  while (true) {
    Qualifiers Q = CT.getQualifiers();
    CT = CT.getUnqualifiedType();
    const Type *T = CT.getTypePtr();
    unsigned qVal = 0;
    if (Q.hasConst())
      qVal |= 0x1;
    if (Q.hasVolatile())
      qVal |= 0x2;
    if (Q.hasRestrict())
      qVal |= 0x4;
    if(qVal)
      COMBINE_HASH(qVal);

    // Hash in ObjC GC qualifiers?

    if (const BuiltinType *BT = dyn_cast<BuiltinType>(T)) {
      return COMBINE_HASH(BT->getKind());
    }
    if (const PointerType *PT = dyn_cast<PointerType>(T)) {
      COMBINE_HASH('*');
      CT = asCanon(PT->getPointeeType());
      continue;
    }
    if (const ReferenceType *RT = dyn_cast<ReferenceType>(T)) {
      COMBINE_HASH('&');
      CT = asCanon(RT->getPointeeType());
      continue;
    }
    if (const BlockPointerType *BT = dyn_cast<BlockPointerType>(T)) {
      COMBINE_HASH('B');
      CT = asCanon(BT->getPointeeType());
      continue;
    }
    if (const ObjCObjectPointerType *OPT = dyn_cast<ObjCObjectPointerType>(T)) {
      COMBINE_HASH('*');
      CT = asCanon(OPT->getPointeeType());
      continue;
    }
    if (const TagType *TT = dyn_cast<TagType>(T)) {
      return COMBINE_HASH('$', hash(TT->getDecl()->getCanonicalDecl()));
    }
    if (const ObjCInterfaceType *OIT = dyn_cast<ObjCInterfaceType>(T)) {
      return COMBINE_HASH('$', hash(OIT->getDecl()->getCanonicalDecl()));
    }
    if (const ObjCObjectType *OIT = dyn_cast<ObjCObjectType>(T)) {
      for (auto *Prot : OIT->getProtocols())
        COMBINE_HASH(hash(Prot));
      CT = asCanon(OIT->getBaseType());
      continue;
    }
    if (const TemplateTypeParmType *TTP = dyn_cast<TemplateTypeParmType>(T)) {
      return COMBINE_HASH('t', TTP->getDepth(), TTP->getIndex());
    }
    if (const InjectedClassNameType *InjT = dyn_cast<InjectedClassNameType>(T)) {
      CT = asCanon(InjT->getInjectedSpecializationType().getCanonicalType());
      continue;
    }

    break;
  }

  return COMBINE_HASH(tryCache(CT.getAsOpaquePtr(), CT));
}

hash_code IndexRecordHasher::hash(DeclarationName Name) {
  assert(!Name.isEmpty());
  // Measurements for using cache or not here, showed significant slowdown when
  // using the cache for all DeclarationNames when parsing Cocoa, and minor
  // improvement or no difference for a couple of C++ single translation unit
  // files. So we avoid caching DeclarationNames.
  return hashImpl(Name);
}

hash_code IndexRecordHasher::hash(const NestedNameSpecifier *NNS) {
  assert(NNS);
  // Measurements for the C++ single translation unit files did not show much
  // difference here; choosing to cache them currently.
  return tryCache(NNS, NNS);
}

template <typename T>
hash_code IndexRecordHasher::tryCache(const void *Ptr, T Obj) {
  auto It = HashByPtr.find(Ptr);
  if (It != HashByPtr.end())
    return It->second;

  hash_code Hash = hashImpl(Obj);
  // hashImpl() may call into tryCache recursively and mutate
  // HashByPtr, so we use find() earlier and insert the hash with another
  // lookup here instead of calling insert() earlier and utilizing the iterator
  // that insert() returns.
  HashByPtr[Ptr] = Hash;
  return Hash;
}

hash_code IndexRecordHasher::hashImpl(const Decl *D) {
  return DeclHashVisitor(*this).Visit(D);
}

static hash_code computeHash(const IdentifierInfo *II) {
  return hash_value(II->getName());
}

static hash_code computeHash(Selector Sel) {
  unsigned N = Sel.getNumArgs();
  if (N == 0)
    ++N;
  hash_code Hash = INITIAL_HASH;
  for (unsigned I = 0; I != N; ++I)
    if (IdentifierInfo *II = Sel.getIdentifierInfoForSlot(I))
      COMBINE_HASH(computeHash(II));
  return Hash;
}

static hash_code computeHash(TemplateName Name, IndexRecordHasher &Hasher) {
  hash_code Hash = INITIAL_HASH;
  if (TemplateDecl *Template = Name.getAsTemplateDecl()) {
    if (TemplateTemplateParmDecl *TTP
        = dyn_cast<TemplateTemplateParmDecl>(Template)) {
      return COMBINE_HASH('t', TTP->getDepth(), TTP->getIndex());
    }

    return COMBINE_HASH(Hasher.hash(Template->getCanonicalDecl()));
  }

  // FIXME: Hash dependent template names.
  return Hash;
}

static hash_code computeHash(const TemplateArgument &Arg,
                             IndexRecordHasher &Hasher) {
  hash_code Hash = INITIAL_HASH;

  switch (Arg.getKind()) {
  case TemplateArgument::Null:
    break;

  case TemplateArgument::Declaration:
    COMBINE_HASH(Hasher.hash(Arg.getAsDecl()));
    break;

  case TemplateArgument::NullPtr:
    break;

  case TemplateArgument::TemplateExpansion:
    COMBINE_HASH('P'); // pack expansion of...
    LLVM_FALLTHROUGH;
  case TemplateArgument::Template:
    COMBINE_HASH(computeHash(Arg.getAsTemplateOrTemplatePattern(), Hasher));
    break;
      
  case TemplateArgument::Expression:
    // FIXME: Hash expressions.
    break;
      
  case TemplateArgument::Pack:
    COMBINE_HASH('p');
    for (const auto &P : Arg.pack_elements())
      COMBINE_HASH(computeHash(P, Hasher));
    break;
      
  case TemplateArgument::Type:
    COMBINE_HASH(Hasher.hash(Arg.getAsType()));
    break;
      
  case TemplateArgument::Integral:
    COMBINE_HASH('V', Hasher.hash(Arg.getIntegralType()), Arg.getAsIntegral());
    break;
  }

  return Hash;
}

hash_code IndexRecordHasher::hashImpl(CanQualType CQT) {
  hash_code Hash = INITIAL_HASH;

  auto asCanon = [](QualType Ty) -> CanQualType {
    return CanQualType::CreateUnsafe(Ty);
  };

  const Type *T = CQT.getTypePtr();

  if (const PackExpansionType *Expansion = dyn_cast<PackExpansionType>(T)) {
    return COMBINE_HASH('P', hash(asCanon(Expansion->getPattern())));
  }
  if (const RValueReferenceType *RT = dyn_cast<RValueReferenceType>(T)) {
    return COMBINE_HASH('%', hash(asCanon(RT->getPointeeType())));
  }
  if (const FunctionProtoType *FT = dyn_cast<FunctionProtoType>(T)) {
    COMBINE_HASH('F', hash(asCanon(FT->getReturnType())));
    for (const auto &I : FT->param_types())
      COMBINE_HASH(hash(asCanon(I)));
    return COMBINE_HASH(FT->isVariadic());
  }
  if (const ComplexType *CT = dyn_cast<ComplexType>(T)) {
    return COMBINE_HASH('<', hash(asCanon(CT->getElementType())));
  }
  if (const TemplateSpecializationType *Spec
      = dyn_cast<TemplateSpecializationType>(T)) {
    COMBINE_HASH('>', computeHash(Spec->getTemplateName(), *this));
    for (unsigned I = 0, N = Spec->getNumArgs(); I != N; ++I)
      COMBINE_HASH(computeHash(Spec->getArg(I), *this));
    return Hash;
  }
  if (const DependentNameType *DNT = dyn_cast<DependentNameType>(T)) {
    COMBINE_HASH('^');
    if (const NestedNameSpecifier *NNS = DNT->getQualifier())
      COMBINE_HASH(hash(NNS));
    return COMBINE_HASH(computeHash(DNT->getIdentifier()));
  }

  // Unhandled type.
  return Hash;
}

hash_code IndexRecordHasher::hashImpl(DeclarationName Name) {
  hash_code Hash = INITIAL_HASH;
  COMBINE_HASH(Name.getNameKind());

  switch (Name.getNameKind()) {
    case DeclarationName::Identifier:
      COMBINE_HASH(computeHash(Name.getAsIdentifierInfo()));
      break;
    case DeclarationName::ObjCZeroArgSelector:
    case DeclarationName::ObjCOneArgSelector:
    case DeclarationName::ObjCMultiArgSelector:
      COMBINE_HASH(computeHash(Name.getObjCSelector()));
      break;
    case DeclarationName::CXXConstructorName:
    case DeclarationName::CXXDestructorName:
    case DeclarationName::CXXConversionFunctionName:
      break;
    case DeclarationName::CXXOperatorName:
      COMBINE_HASH(Name.getCXXOverloadedOperator());
      break;
    case DeclarationName::CXXLiteralOperatorName:
      COMBINE_HASH(computeHash(Name.getCXXLiteralIdentifier()));
      break;
    case DeclarationName::CXXUsingDirective:
      break;
    case DeclarationName::CXXDeductionGuideName:
      COMBINE_HASH(computeHash(Name.getCXXDeductionGuideTemplate()
                 ->getDeclName().getAsIdentifierInfo()));
      break;
  }

  return Hash;
}

hash_code IndexRecordHasher::hashImpl(const NestedNameSpecifier *NNS) {
  hash_code Hash = INITIAL_HASH;
  if (auto *Pre = NNS->getPrefix())
    COMBINE_HASH(hash(Pre));

  COMBINE_HASH(NNS->getKind());

  switch (NNS->getKind()) {
  case NestedNameSpecifier::Identifier:
    COMBINE_HASH(computeHash(NNS->getAsIdentifier()));
    break;

  case NestedNameSpecifier::Namespace:
    COMBINE_HASH(hash(NNS->getAsNamespace()->getCanonicalDecl()));
    break;

  case NestedNameSpecifier::NamespaceAlias:
    COMBINE_HASH(hash(NNS->getAsNamespaceAlias()->getCanonicalDecl()));
    break;

  case NestedNameSpecifier::Global:
    break;

  case NestedNameSpecifier::Super:
    break;

  case NestedNameSpecifier::TypeSpecWithTemplate:
    // Fall through to hash the type.

  case NestedNameSpecifier::TypeSpec:
    COMBINE_HASH(hash(QualType(NNS->getAsType(), 0)));
    break;
  }

  return Hash;
}
