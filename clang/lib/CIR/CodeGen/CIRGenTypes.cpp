#include "CIRGenTypes.h"

#include "CIRGenModule.h"

#include "clang/AST/ASTContext.h"
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

mlir::Type CIRGenTypes::ConvertFunctionTypeInternal(QualType qft) {
  assert(qft.isCanonical());
  const FunctionType *ft = cast<FunctionType>(qft.getTypePtr());
  // First, check whether we can build the full fucntion type. If the function
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

mlir::Type CIRGenTypes::convertType(QualType type) {
  type = astContext.getCanonicalType(type);
  const Type *ty = type.getTypePtr();

  // Has the type already been processed?
  TypeCacheTy::iterator tci = typeCache.find(ty);
  if (tci != typeCache.end())
    return tci->second;

  // For types that haven't been implemented yet or are otherwise unsupported,
  // report an error and return 'int'.

  mlir::Type resultType = nullptr;
  switch (ty->getTypeClass()) {
  case Type::Builtin: {
    switch (cast<BuiltinType>(ty)->getKind()) {

    // void
    case BuiltinType::Void:
      resultType = cgm.VoidTy;
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

  case Type::FunctionNoProto:
  case Type::FunctionProto:
    resultType = ConvertFunctionTypeInternal(type);
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
    cgm.errorNYI(SourceLocation(), "processing of type", type);
    resultType = cgm.SInt32Ty;
    break;
  }

  assert(resultType && "Type conversion not yet implemented");

  typeCache[ty] = resultType;
  return resultType;
}
