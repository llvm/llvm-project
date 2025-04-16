#include "CIRGenTypes.h"

#include "CIRGenModule.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Type.h"
#include "clang/Basic/TargetInfo.h"

#include <cassert>

using namespace clang;
using namespace clang::CIRGen;

CIRGenTypes::CIRGenTypes(CIRGenModule &genModule)
    : cgm(genModule), astContext(genModule.getASTContext()),
      builder(cgm.getBuilder()) {}

CIRGenTypes::~CIRGenTypes() {}

mlir::MLIRContext &CIRGenTypes::getMLIRContext() const {
  return *builder.getContext();
}

/// Return true if the specified type in a function parameter or result position
/// can be converted to a CIR type at this point. This boils down to being
/// whether it is complete, as well as whether we've temporarily deferred
/// expanding the type because we're in a recursive context.
bool CIRGenTypes::isFuncParamTypeConvertible(clang::QualType type) {
  // Some ABIs cannot have their member pointers represented in LLVM IR unless
  // certain circumstances have been reached.
  assert(!type->getAs<MemberPointerType>() && "NYI");

  // If this isn't a tag type, we can convert it.
  const TagType *tagType = type->getAs<TagType>();
  if (!tagType)
    return true;

  // Function types involving incomplete class types are problematic in MLIR.
  return !tagType->isIncompleteType();
}

/// Code to verify a given function type is complete, i.e. the return type and
/// all of the parameter types are complete. Also check to see if we are in a
/// RS_StructPointer context, and if so whether any struct types have been
/// pended. If so, we don't want to ask the ABI lowering code to handle a type
/// that cannot be converted to a CIR type.
bool CIRGenTypes::isFuncTypeConvertible(const FunctionType *ft) {
  if (!isFuncParamTypeConvertible(ft->getReturnType()))
    return false;

  if (const auto *fpt = dyn_cast<FunctionProtoType>(ft))
    for (unsigned i = 0, e = fpt->getNumParams(); i != e; i++)
      if (!isFuncParamTypeConvertible(fpt->getParamType(i)))
        return false;

  return true;
}

mlir::Type CIRGenTypes::convertFunctionTypeInternal(QualType qft) {
  assert(qft.isCanonical());
  const FunctionType *ft = cast<FunctionType>(qft.getTypePtr());
  // First, check whether we can build the full function type. If the function
  // type depends on an incomplete type (e.g. a struct or enum), we cannot lower
  // the function type.
  if (!isFuncTypeConvertible(ft)) {
    cgm.errorNYI(SourceLocation(), "function type involving an incomplete type",
                 qft);
    return cir::FuncType::get(SmallVector<mlir::Type, 1>{}, cgm.VoidTy);
  }

  // TODO(CIR): This is a stub of what the final code will be.  See the
  // implementation of this function and the implementation of class
  // CIRGenFunction in the ClangIR incubator project.

  if (const auto *fpt = dyn_cast<FunctionProtoType>(ft)) {
    SmallVector<mlir::Type> mlirParamTypes;
    for (unsigned i = 0; i < fpt->getNumParams(); ++i) {
      mlirParamTypes.push_back(convertType(fpt->getParamType(i)));
    }
    return cir::FuncType::get(
        mlirParamTypes, convertType(fpt->getReturnType().getUnqualifiedType()),
        fpt->isVariadic());
  }
  cgm.errorNYI(SourceLocation(), "non-prototype function type", qft);
  return cir::FuncType::get(SmallVector<mlir::Type, 1>{}, cgm.VoidTy);
}

// This is CIR's version of CodeGenTypes::addRecordTypeName. It isn't shareable
// because CIR has different uniquing requirements.
std::string CIRGenTypes::getRecordTypeName(const clang::RecordDecl *recordDecl,
                                           StringRef suffix) {
  llvm::SmallString<256> typeName;
  llvm::raw_svector_ostream outStream(typeName);

  PrintingPolicy policy = recordDecl->getASTContext().getPrintingPolicy();
  policy.SuppressInlineNamespace = false;
  policy.AlwaysIncludeTypeForTemplateArgument = true;
  policy.PrintAsCanonical = true;
  policy.SuppressTagKeyword = true;

  if (recordDecl->getIdentifier())
    astContext.getRecordType(recordDecl).print(outStream, policy);
  else if (auto *typedefNameDecl = recordDecl->getTypedefNameForAnonDecl())
    typedefNameDecl->printQualifiedName(outStream, policy);
  else
    outStream << builder.getUniqueAnonRecordName();

  if (!suffix.empty())
    outStream << suffix;

  return builder.getUniqueRecordName(std::string(typeName));
}

/// Lay out a tagged decl type like struct or union.
mlir::Type CIRGenTypes::convertRecordDeclType(const clang::RecordDecl *rd) {
  // TagDecl's are not necessarily unique, instead use the (clang) type
  // connected to the decl.
  const Type *key = astContext.getTagDeclType(rd).getTypePtr();
  cir::RecordType entry = recordDeclTypes[key];

  // If we don't have an entry for this record yet, create one.
  // We create an incomplete type initially. If `rd` is complete, we will
  // add the members below.
  if (!entry) {
    auto name = getRecordTypeName(rd, "");
    entry = builder.getIncompleteRecordTy(name, rd);
    recordDeclTypes[key] = entry;
  }

  rd = rd->getDefinition();
  if (!rd || !rd->isCompleteDefinition() || entry.isComplete())
    return entry;

  cgm.errorNYI(rd->getSourceRange(), "Complete record type");
  return entry;
}

mlir::Type CIRGenTypes::convertType(QualType type) {
  type = astContext.getCanonicalType(type);
  const Type *ty = type.getTypePtr();

  // Process record types before the type cache lookup.
  if (const auto *recordType = dyn_cast<RecordType>(type))
    return convertRecordDeclType(recordType->getDecl());

  // Has the type already been processed?
  TypeCacheTy::iterator tci = typeCache.find(ty);
  if (tci != typeCache.end())
    return tci->second;

  // For types that haven't been implemented yet or are otherwise unsupported,
  // report an error and return 'int'.

  mlir::Type resultType = nullptr;
  switch (ty->getTypeClass()) {
  case Type::Record:
    llvm_unreachable("Should have been handled above");

  case Type::Builtin: {
    switch (cast<BuiltinType>(ty)->getKind()) {
    // void
    case BuiltinType::Void:
      resultType = cgm.VoidTy;
      break;

    // bool
    case BuiltinType::Bool:
      resultType = cir::BoolType::get(&getMLIRContext());
      break;

    // Signed integral types.
    case BuiltinType::Char_S:
    case BuiltinType::Int:
    case BuiltinType::Int128:
    case BuiltinType::Long:
    case BuiltinType::LongLong:
    case BuiltinType::SChar:
    case BuiltinType::Short:
    case BuiltinType::WChar_S:
      resultType =
          cir::IntType::get(&getMLIRContext(), astContext.getTypeSize(ty),
                            /*isSigned=*/true);
      break;
    // Unsigned integral types.
    case BuiltinType::Char8:
    case BuiltinType::Char16:
    case BuiltinType::Char32:
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
    case BuiltinType::UInt:
    case BuiltinType::UInt128:
    case BuiltinType::ULong:
    case BuiltinType::ULongLong:
    case BuiltinType::UShort:
    case BuiltinType::WChar_U:
      resultType =
          cir::IntType::get(&getMLIRContext(), astContext.getTypeSize(ty),
                            /*isSigned=*/false);
      break;

    // Floating-point types
    case BuiltinType::Float16:
      resultType = cgm.FP16Ty;
      break;
    case BuiltinType::Half:
      if (astContext.getLangOpts().NativeHalfType ||
          !astContext.getTargetInfo().useFP16ConversionIntrinsics()) {
        resultType = cgm.FP16Ty;
      } else {
        cgm.errorNYI(SourceLocation(), "processing of built-in type", type);
        resultType = cgm.SInt32Ty;
      }
      break;
    case BuiltinType::BFloat16:
      resultType = cgm.BFloat16Ty;
      break;
    case BuiltinType::Float:
      assert(&astContext.getFloatTypeSemantics(type) ==
                 &llvm::APFloat::IEEEsingle() &&
             "ClangIR NYI: 'float' in a format other than IEEE 32-bit");
      resultType = cgm.FloatTy;
      break;
    case BuiltinType::Double:
      assert(&astContext.getFloatTypeSemantics(type) ==
                 &llvm::APFloat::IEEEdouble() &&
             "ClangIR NYI: 'double' in a format other than IEEE 64-bit");
      resultType = cgm.DoubleTy;
      break;
    case BuiltinType::LongDouble:
      resultType =
          builder.getLongDoubleTy(astContext.getFloatTypeSemantics(type));
      break;
    case BuiltinType::Float128:
      resultType = cgm.FP128Ty;
      break;
    case BuiltinType::Ibm128:
      cgm.errorNYI(SourceLocation(), "processing of built-in type", type);
      resultType = cgm.SInt32Ty;
      break;

    case BuiltinType::NullPtr:
      // Add proper CIR type for it? this looks mostly useful for sema related
      // things (like for overloads accepting void), for now, given that
      // `sizeof(std::nullptr_t)` is equal to `sizeof(void *)`, model
      // std::nullptr_t as !cir.ptr<!void>
      resultType = builder.getVoidPtrTy();
      break;

    default:
      cgm.errorNYI(SourceLocation(), "processing of built-in type", type);
      resultType = cgm.SInt32Ty;
      break;
    }
    break;
  }

  case Type::Pointer: {
    const PointerType *ptrTy = cast<PointerType>(ty);
    QualType elemTy = ptrTy->getPointeeType();
    assert(!elemTy->isConstantMatrixType() && "not implemented");

    mlir::Type pointeeType = convertType(elemTy);

    resultType = builder.getPointerTo(pointeeType);
    break;
  }

  case Type::ConstantArray: {
    const ConstantArrayType *arrTy = cast<ConstantArrayType>(ty);
    mlir::Type elemTy = convertTypeForMem(arrTy->getElementType());
    resultType = cir::ArrayType::get(builder.getContext(), elemTy,
                                     arrTy->getSize().getZExtValue());
    break;
  }

  case Type::FunctionNoProto:
  case Type::FunctionProto:
    resultType = convertFunctionTypeInternal(type);
    break;

  case Type::BitInt: {
    const auto *bitIntTy = cast<BitIntType>(type);
    if (bitIntTy->getNumBits() > cir::IntType::maxBitwidth()) {
      cgm.errorNYI(SourceLocation(), "large _BitInt type", type);
      resultType = cgm.SInt32Ty;
    } else {
      resultType = cir::IntType::get(&getMLIRContext(), bitIntTy->getNumBits(),
                                     bitIntTy->isSigned());
    }
    break;
  }

  default:
    cgm.errorNYI(SourceLocation(), "processing of type",
                 type->getTypeClassName());
    resultType = cgm.SInt32Ty;
    break;
  }

  assert(resultType && "Type conversion not yet implemented");

  typeCache[ty] = resultType;
  return resultType;
}

mlir::Type CIRGenTypes::convertTypeForMem(clang::QualType qualType,
                                          bool forBitField) {
  assert(!qualType->isConstantMatrixType() && "Matrix types NYI");

  mlir::Type convertedType = convertType(qualType);

  assert(!forBitField && "Bit fields NYI");

  // If this is a bit-precise integer type in a bitfield representation, map
  // this integer to the target-specified size.
  if (forBitField && qualType->isBitIntType())
    assert(!qualType->isBitIntType() && "Bit field with type _BitInt NYI");

  return convertedType;
}

bool CIRGenTypes::isZeroInitializable(clang::QualType t) {
  if (t->getAs<PointerType>())
    return astContext.getTargetNullPointerValue(t) == 0;

  if (const auto *at = astContext.getAsArrayType(t)) {
    if (isa<IncompleteArrayType>(at))
      return true;

    if (const auto *cat = dyn_cast<ConstantArrayType>(at))
      if (astContext.getConstantArrayElementCount(cat) == 0)
        return true;
  }

  if (t->getAs<RecordType>()) {
    cgm.errorNYI(SourceLocation(), "isZeroInitializable for RecordType", t);
    return false;
  }

  if (t->getAs<MemberPointerType>()) {
    cgm.errorNYI(SourceLocation(), "isZeroInitializable for MemberPointerType",
                 t);
    return false;
  }

  return true;
}

const CIRGenFunctionInfo &CIRGenTypes::arrangeCIRFunctionInfo() {
  // Lookup or create unique function info.
  llvm::FoldingSetNodeID id;
  CIRGenFunctionInfo::Profile(id);

  void *insertPos = nullptr;
  CIRGenFunctionInfo *fi = functionInfos.FindNodeOrInsertPos(id, insertPos);
  if (fi)
    return *fi;

  assert(!cir::MissingFeatures::opCallCallConv());

  // Construction the function info. We co-allocate the ArgInfos.
  fi = CIRGenFunctionInfo::create();
  functionInfos.InsertNode(fi, insertPos);

  bool inserted = functionsBeingProcessed.insert(fi).second;
  (void)inserted;
  assert(inserted && "Are functions being processed recursively?");

  assert(!cir::MissingFeatures::opCallCallConv());
  assert(!cir::MissingFeatures::opCallArgs());

  bool erased = functionsBeingProcessed.erase(fi);
  (void)erased;
  assert(erased && "Not in set?");

  return *fi;
}
