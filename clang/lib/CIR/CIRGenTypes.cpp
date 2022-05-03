#include "CIRGenTypes.h"
#include "CIRGenCall.h"
#include "CIRGenFunctionInfo.h"
#include "CIRGenModule.h"
#include "CallingConv.h"
#include "TargetInfo.h"

#include "mlir/Dialect/CIR/IR/CIRTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/RecordLayout.h"

using namespace clang;
using namespace cir;

unsigned CIRGenTypes::ClangCallConvToCIRCallConv(clang::CallingConv CC) {
  assert(CC == CC_C && "No other calling conventions implemented.");
  return cir::CallingConv::C;
}

CIRGenTypes::CIRGenTypes(CIRGenModule &cgm)
    : Context(cgm.getASTContext()), Builder(cgm.getBuilder()), CGM{cgm},
      Target(cgm.getTarget()), TheCXXABI(cgm.getCXXABI()),
      TheABIInfo(cgm.getTargetCIRGenInfo().getABIInfo()) {}
CIRGenTypes::~CIRGenTypes() {
  for (llvm::FoldingSet<CIRGenFunctionInfo>::iterator I = FunctionInfos.begin(),
                                                      E = FunctionInfos.end();
       I != E;)
    delete &*I++;
}

std::string CIRGenTypes::getRecordTypeName(const clang::RecordDecl *recordDecl,
                                           StringRef suffix) {
  llvm::SmallString<256> typeName;
  llvm::raw_svector_ostream outStream(typeName);

  outStream << recordDecl->getKindName() << '.';

  PrintingPolicy policy = recordDecl->getASTContext().getPrintingPolicy();
  policy.SuppressInlineNamespace = false;

  if (recordDecl->getIdentifier()) {
    if (recordDecl->getDeclContext())
      recordDecl->printQualifiedName(outStream, policy);
    else
      recordDecl->DeclaratorDecl::printName(outStream);
  } else if (auto *typedefNameDecl = recordDecl->getTypedefNameForAnonDecl()) {
    if (typedefNameDecl->getDeclContext())
      typedefNameDecl->printQualifiedName(outStream, policy);
    else
      typedefNameDecl->printName(outStream);
  } else {
    outStream << "anon";
  }

  if (!suffix.empty())
    outStream << suffix;

  return std::string(typeName);
}

mlir::Type
CIRGenTypes::convertRecordDeclType(const clang::RecordDecl *recordDecl) {
  const auto *key = Context.getTagDeclType(recordDecl).getTypePtr();
  mlir::cir::StructType &entry = recordDeclTypes[key];

  recordDecl = recordDecl->getDefinition();
  // TODO: clang checks here whether the type is known to be opaque. This is
  // equivalent to a forward decl. Is checking for a non-null entry close enough
  // of a match?
  if (!recordDecl || !recordDecl->isCompleteDefinition() || entry)
    return entry;

  // TODO: Implement checking for whether or not this type is safe to convert.

  // TODO: handle whether or not layout was skipped and recursive record layout

  if (const auto *cxxRecordDecl = dyn_cast<CXXRecordDecl>(recordDecl)) {
    assert(cxxRecordDecl->bases().begin() == cxxRecordDecl->bases().end() &&
           "Base clases NYI");
  }

  entry = computeRecordLayout(recordDecl);

  return entry;
}

mlir::Type CIRGenTypes::convertTypeForMem(clang::QualType qualType,
                                          bool forBitField) {
  assert(!qualType->isConstantMatrixType() && "Matrix types NYI");

  mlir::Type convertedType = ConvertType(qualType);

  assert(!forBitField && "Bit fields NYI");
  assert(!qualType->isBitIntType() && "BitIntType NYI");

  return convertedType;
}

mlir::MLIRContext &CIRGenTypes::getMLIRContext() const {
  return *Builder.getContext();
}

mlir::Type CIRGenTypes::ConvertFunctionTypeInternal(QualType QFT) {
  assert(QFT.isCanonical());
  const Type *Ty = QFT.getTypePtr();
  const FunctionType *FT = cast<FunctionType>(QFT.getTypePtr());
  // First, check whether we can build the full fucntion type. If the function
  // type depends on an incomplete type (e.g. a struct or enum), we cannot lower
  // the function type.
  assert(isFuncTypeConvertible(FT) && "NYI");

  // While we're converting the parameter types for a function, we don't want to
  // recursively convert any pointed-to structs. Converting directly-used
  // structs is ok though.
  assert(RecordsBeingLaidOut.insert(Ty).second && "NYI");

  // The function type can be built; call the appropriate routines to build it
  const CIRGenFunctionInfo *FI;
  const auto *FPT = dyn_cast<FunctionProtoType>(FT);
  assert(FPT && "FunctionNonPrototype NIY");
  FI = &arrangeFreeFunctionType(
      CanQual<FunctionProtoType>::CreateUnsafe(QualType(FPT, 0)));

  mlir::Type ResultType = nullptr;
  // If there is something higher level prodding our CIRGenFunctionInfo, then
  // don't recurse into it again.
  assert(!FunctionsBeingProcessed.count(FI) && "NYI");

  // Otherwise, we're good to go, go ahead and convert it.
  ResultType = GetFunctionType(*FI);

  RecordsBeingLaidOut.erase(Ty);

  assert(!SkippedLayout && "Shouldn't have skipped anything yet");

  assert(RecordsBeingLaidOut.empty() && "Deferral NYI");
  assert(DeferredRecords.empty() && "Deferral NYI");

  return ResultType;
}

/// isFuncParamTypeConvertible - Return true if the specified type in a function
/// parameter or result position can be converted to a CIR type at this point.
/// This boils down to being whether it is complete, as well as whether we've
/// temporarily deferred expanding the type because we're in a recursive
/// context.
bool CIRGenTypes::isFuncParamTypeConvertible(clang::QualType Ty) {
  // Some ABIs cannot have their member pointers represented in LLVM IR unless
  // certain circumstances have been reached.
  assert(!Ty->getAs<MemberPointerType>() && "NYI");

  // If this isn't a tagged type, we can convert it!
  auto *TT = Ty->getAs<TagType>();
  assert(!TT && "Only non-TagTypes implemented atm.");
  return true;
}

/// Code to verify a given function type is complete, i.e. the return type and
/// all of the parameter types are complete. Also check to see if we are in a
/// RS_StructPointer context, and if so whether any struct types have been
/// pended. If so, we don't want to ask the ABI lowering code to handle a type
/// that cannot be converted to a CIR type.
bool CIRGenTypes::isFuncTypeConvertible(const FunctionType *FT) {
  if (!isFuncParamTypeConvertible(FT->getReturnType()))
    return false;

  if (const auto *FPT = dyn_cast<FunctionProtoType>(FT))
    for (unsigned i = 0, e = FPT->getNumParams(); i != e; i++)
      if (!isFuncParamTypeConvertible(FPT->getParamType(i)))
        return false;

  return true;
}

/// ConvertType - Convert the specified type to its MLIR form.
mlir::Type CIRGenTypes::ConvertType(QualType T) {
  T = Context.getCanonicalType(T);
  const Type *Ty = T.getTypePtr();

  // For the device-side compilation, CUDA device builtin surface/texture types
  // may be represented in different types.
  assert(!Context.getLangOpts().CUDAIsDevice && "not implemented");

  if (const auto *recordType = dyn_cast<RecordType>(T))
    return convertRecordDeclType(recordType->getDecl());

  // See if type is already cached.
  TypeCacheTy::iterator TCI = TypeCache.find(Ty);
  // If type is found in map then use it. Otherwise, convert type T.
  if (TCI != TypeCache.end())
    return TCI->second;

  // If we don't have it in the cache, convert it now.
  mlir::Type ResultType = nullptr;
  switch (Ty->getTypeClass()) {
  case Type::Record: // Handled above.
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.inc"
    llvm_unreachable("Non-canonical or dependent types aren't possible.");

  case Type::ArrayParameter:
    llvm_unreachable("NYI");

  case Type::Builtin: {
    switch (cast<BuiltinType>(Ty)->getKind()) {
    case BuiltinType::WasmExternRef:
    case BuiltinType::SveBoolx2:
    case BuiltinType::SveBoolx4:
    case BuiltinType::SveCount:
      llvm_unreachable("NYI");
    case BuiltinType::Void:
    case BuiltinType::ObjCId:
    case BuiltinType::ObjCClass:
    case BuiltinType::ObjCSel:
      // FIXME: if we emit like LLVM we probably wanna use i8.
      assert(0 && "not implemented");
      break;

    case BuiltinType::Bool:
      ResultType = ::mlir::cir::BoolType::get(Builder.getContext());
      break;

    case BuiltinType::Char_S:
    case BuiltinType::Char_U:
    case BuiltinType::SChar:
    case BuiltinType::UChar:
    case BuiltinType::Short:
    case BuiltinType::UShort:
    case BuiltinType::Int:
    case BuiltinType::UInt:
    case BuiltinType::Long:
    case BuiltinType::ULong:
    case BuiltinType::LongLong:
    case BuiltinType::ULongLong:
    case BuiltinType::WChar_S:
    case BuiltinType::WChar_U:
    case BuiltinType::Char8:
    case BuiltinType::Char16:
    case BuiltinType::Char32:
    case BuiltinType::ShortAccum:
    case BuiltinType::Accum:
    case BuiltinType::LongAccum:
    case BuiltinType::UShortAccum:
    case BuiltinType::UAccum:
    case BuiltinType::ULongAccum:
    case BuiltinType::ShortFract:
    case BuiltinType::Fract:
    case BuiltinType::LongFract:
    case BuiltinType::UShortFract:
    case BuiltinType::UFract:
    case BuiltinType::ULongFract:
    case BuiltinType::SatShortAccum:
    case BuiltinType::SatAccum:
    case BuiltinType::SatLongAccum:
    case BuiltinType::SatUShortAccum:
    case BuiltinType::SatUAccum:
    case BuiltinType::SatULongAccum:
    case BuiltinType::SatShortFract:
    case BuiltinType::SatFract:
    case BuiltinType::SatLongFract:
    case BuiltinType::SatUShortFract:
    case BuiltinType::SatUFract:
    case BuiltinType::SatULongFract:
      // FIXME: break this in s/u and also pass signed param.
      ResultType =
          Builder.getIntegerType(static_cast<unsigned>(Context.getTypeSize(T)));
      break;

    case BuiltinType::Float16:
      ResultType = Builder.getF16Type();
      break;
    case BuiltinType::Half:
      // Should be the same as above?
      assert(0 && "not implemented");
      break;
    case BuiltinType::BFloat16:
      ResultType = Builder.getBF16Type();
      break;
    case BuiltinType::Float:
      ResultType = Builder.getF32Type();
      break;
    case BuiltinType::Double:
      ResultType = Builder.getF64Type();
      break;
    case BuiltinType::LongDouble:
    case BuiltinType::Float128:
    case BuiltinType::Ibm128:
      // FIXME: look at Context.getFloatTypeSemantics(T) and getTypeForFormat
      // on LLVM codegen.
      assert(0 && "not implemented");
      break;

    case BuiltinType::NullPtr:
      // Model std::nullptr_t as i8*
      // ResultType = llvm::Type::getInt8PtrTy(getLLVMContext());
      assert(0 && "not implemented");
      break;

    case BuiltinType::UInt128:
    case BuiltinType::Int128:
      assert(0 && "not implemented");
      // FIXME: ResultType = Builder.getIntegerType(128);
      break;

#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix)                   \
  case BuiltinType::Id:
#include "clang/Basic/OpenCLImageTypes.def"
#define EXT_OPAQUE_TYPE(ExtType, Id, Ext) case BuiltinType::Id:
#include "clang/Basic/OpenCLExtensionTypes.def"
    case BuiltinType::OCLSampler:
    case BuiltinType::OCLEvent:
    case BuiltinType::OCLClkEvent:
    case BuiltinType::OCLQueue:
    case BuiltinType::OCLReserveID:
      assert(0 && "not implemented");
      break;
    case BuiltinType::SveInt8:
    case BuiltinType::SveUint8:
    case BuiltinType::SveInt8x2:
    case BuiltinType::SveUint8x2:
    case BuiltinType::SveInt8x3:
    case BuiltinType::SveUint8x3:
    case BuiltinType::SveInt8x4:
    case BuiltinType::SveUint8x4:
    case BuiltinType::SveInt16:
    case BuiltinType::SveUint16:
    case BuiltinType::SveInt16x2:
    case BuiltinType::SveUint16x2:
    case BuiltinType::SveInt16x3:
    case BuiltinType::SveUint16x3:
    case BuiltinType::SveInt16x4:
    case BuiltinType::SveUint16x4:
    case BuiltinType::SveInt32:
    case BuiltinType::SveUint32:
    case BuiltinType::SveInt32x2:
    case BuiltinType::SveUint32x2:
    case BuiltinType::SveInt32x3:
    case BuiltinType::SveUint32x3:
    case BuiltinType::SveInt32x4:
    case BuiltinType::SveUint32x4:
    case BuiltinType::SveInt64:
    case BuiltinType::SveUint64:
    case BuiltinType::SveInt64x2:
    case BuiltinType::SveUint64x2:
    case BuiltinType::SveInt64x3:
    case BuiltinType::SveUint64x3:
    case BuiltinType::SveInt64x4:
    case BuiltinType::SveUint64x4:
    case BuiltinType::SveBool:
    case BuiltinType::SveFloat16:
    case BuiltinType::SveFloat16x2:
    case BuiltinType::SveFloat16x3:
    case BuiltinType::SveFloat16x4:
    case BuiltinType::SveFloat32:
    case BuiltinType::SveFloat32x2:
    case BuiltinType::SveFloat32x3:
    case BuiltinType::SveFloat32x4:
    case BuiltinType::SveFloat64:
    case BuiltinType::SveFloat64x2:
    case BuiltinType::SveFloat64x3:
    case BuiltinType::SveFloat64x4:
    case BuiltinType::SveBFloat16:
    case BuiltinType::SveBFloat16x2:
    case BuiltinType::SveBFloat16x3:
    case BuiltinType::SveBFloat16x4: {
      assert(0 && "not implemented");
      break;
    }
#define PPC_VECTOR_TYPE(Name, Id, Size)                                        \
  case BuiltinType::Id:                                                        \
    assert(0 && "not implemented");                                            \
    break;
#include "clang/Basic/PPCTypes.def"
#define RVV_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/RISCVVTypes.def"
      {
        assert(0 && "not implemented");
        break;
      }
    case BuiltinType::Dependent:
#define BUILTIN_TYPE(Id, SingletonId)
#define PLACEHOLDER_TYPE(Id, SingletonId) case BuiltinType::Id:
#include "clang/AST/BuiltinTypes.def"
      llvm_unreachable("Unexpected placeholder builtin type!");
    }
    break;
  }
  case Type::Auto:
  case Type::DeducedTemplateSpecialization:
    llvm_unreachable("Unexpected undeduced type!");
  case Type::Complex: {
    assert(0 && "not implemented");
    break;
  }
  case Type::LValueReference:
  case Type::RValueReference: {
    assert(0 && "not implemented");
    break;
  }
  case Type::Pointer: {
    const PointerType *PTy = cast<PointerType>(Ty);
    QualType ETy = PTy->getPointeeType();
    assert(!ETy->isConstantMatrixType() && "not implemented");

    mlir::Type PointeeType = ConvertType(ETy);

    // Treat effectively as a *i8.
    // if (PointeeType->isVoidTy())
    //  PointeeType = Builder.getI8Type();

    // FIXME: add address specifier to cir::PointerType?
    ResultType =
        ::mlir::cir::PointerType::get(Builder.getContext(), PointeeType);
    assert(ResultType && "Cannot get pointer type?");
    break;
  }

  case Type::VariableArray: {
    assert(0 && "not implemented");
    break;
  }
  case Type::IncompleteArray: {
    assert(0 && "not implemented");
    break;
  }
  case Type::ConstantArray: {
    const ConstantArrayType *A = cast<ConstantArrayType>(Ty);
    auto EltTy = convertTypeForMem(A->getElementType());

    auto isSized = [&](mlir::Type ty) {
      if (ty.isIntOrFloat() ||
          ty.isa<mlir::cir::PointerType, mlir::cir::StructType,
                 mlir::cir::ArrayType>())
        return true;
      assert(0 && "not implemented");
      return false;
    };

    // FIXME: In LLVM, "lower arrays of undefined struct type to arrays of
    // i8 just to have a concrete type". Not sure this makes sense in CIR yet.
    assert(isSized(EltTy) && "not implemented");
    ResultType = ::mlir::cir::ArrayType::get(Builder.getContext(), EltTy,
                                             A->getSize().getZExtValue());
    break;
  }
  case Type::ExtVector:
  case Type::Vector: {
    assert(0 && "not implemented");
    break;
  }
  case Type::ConstantMatrix: {
    assert(0 && "not implemented");
    break;
  }
  case Type::FunctionNoProto:
  case Type::FunctionProto:
    ResultType = ConvertFunctionTypeInternal(T);
    break;
  case Type::ObjCObject:
    assert(0 && "not implemented");
    break;

  case Type::ObjCInterface: {
    assert(0 && "not implemented");
    break;
  }

  case Type::ObjCObjectPointer: {
    assert(0 && "not implemented");
    break;
  }

  case Type::Enum: {
    assert(0 && "not implemented");
    break;
  }

  case Type::BlockPointer: {
    assert(0 && "not implemented");
    break;
  }

  case Type::MemberPointer: {
    assert(0 && "not implemented");
    break;
  }

  case Type::Atomic: {
    assert(0 && "not implemented");
    break;
  }
  case Type::Pipe: {
    assert(0 && "not implemented");
    break;
  }
  case Type::BitInt: {
    assert(0 && "not implemented");
    break;
  }
  }

  assert(ResultType && "Didn't convert a type?");

  TypeCache[Ty] = ResultType;
  return ResultType;
}

const CIRGenFunctionInfo &CIRGenTypes::arrangeCIRFunctionInfo(
    CanQualType resultType, bool instanceMethod, bool chainCall,
    llvm::ArrayRef<CanQualType> argTypes, FunctionType::ExtInfo info,
    llvm::ArrayRef<FunctionProtoType::ExtParameterInfo> paramInfos,
    RequiredArgs required) {
  assert(llvm::all_of(argTypes,
                      [](CanQualType T) { return T.isCanonicalAsParam(); }));

  // Lookup or create unique function info.
  llvm::FoldingSetNodeID ID;
  CIRGenFunctionInfo::Profile(ID, instanceMethod, chainCall, info, paramInfos,
                              required, resultType, argTypes);

  void *insertPos = nullptr;
  CIRGenFunctionInfo *FI = FunctionInfos.FindNodeOrInsertPos(ID, insertPos);
  if (FI)
    return *FI;

  unsigned CC = ClangCallConvToCIRCallConv(info.getCC());

  // Construction the function info. We co-allocate the ArgInfos.
  FI = CIRGenFunctionInfo::create(CC, instanceMethod, chainCall, info,
                                  paramInfos, resultType, argTypes, required);
  FunctionInfos.InsertNode(FI, insertPos);

  bool inserted = FunctionsBeingProcessed.insert(FI).second;
  (void)inserted;
  assert(inserted && "Recursively being processed?");

  // Compute ABI inforamtion.
  assert(info.getCC() != clang::CallingConv::CC_SpirFunction && "NYI");
  assert(info.getCC() != CC_Swift && info.getCC() != CC_SwiftAsync &&
         "Swift NYI");
  getABIInfo().computeInfo(*FI);

  // Loop over all of the computed argument and return value info. If any of
  // them are direct or extend without a specified coerce type, specify the
  // default now.
  ABIArgInfo &retInfo = FI->getReturnInfo();
  if (retInfo.canHaveCoerceToType() && retInfo.getCoerceToType() == nullptr)
    retInfo.setCoerceToType(ConvertType(FI->getReturnType()));

  for (auto &I : FI->arguments())
    if (I.info.canHaveCoerceToType() && I.info.getCoerceToType() == nullptr)
      I.info.setCoerceToType(ConvertType(I.type));

  bool erased = FunctionsBeingProcessed.erase(FI);
  (void)erased;
  assert(erased && "Not in set?");

  return *FI;
}

const CIRGenFunctionInfo &CIRGenTypes::arrangeGlobalDeclaration(GlobalDecl GD) {
  assert(!dyn_cast<ObjCMethodDecl>(GD.getDecl()) &&
         "This is reported as a FIXME in codegen");
  const auto *FD = cast<FunctionDecl>(GD.getDecl());

  assert(!isa<CXXConstructorDecl>(GD.getDecl()) &&
         !isa<CXXDestructorDecl>(GD.getDecl()) && "NYI");

  return arrangeFunctionDeclaration(FD);
}

// UpdateCompletedType - When we find the full definition for a TagDecl,
// replace the 'opaque' type we previously made for it if applicable.
void CIRGenTypes::UpdateCompletedType(const TagDecl *TD) {
  // If this is an enum being completed, then we flush all non-struct types
  // from the cache. This allows function types and other things that may be
  // derived from the enum to be recomputed.
  if (const auto *ED = dyn_cast<EnumDecl>(TD)) {
    llvm_unreachable("NYI");
  }

  // If we completed a RecordDecl that we previously used and converted to an
  // anonymous type, then go ahead and complete it now.
  const auto *RD = cast<RecordDecl>(TD);
  if (RD->isDependentType())
    return;

  // Only complete if we converted it already. If we haven't converted it yet,
  // we'll just do it lazily.
  if (recordDeclTypes.count(Context.getTagDeclType(RD).getTypePtr()))
    convertRecordDeclType(RD);

  // If necessary, provide the full definition of a type only used with a
  // declaration so far.
  if (CGM.getModuleDebugInfo())
    llvm_unreachable("NYI");
}
