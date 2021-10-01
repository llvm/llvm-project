#include "CIRGenTypes.h"

#include "mlir/Dialect/CIR/IR/CIRTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"

using namespace clang;
using namespace cir;

CIRGenTypes::CIRGenTypes(ASTContext &Ctx, mlir::OpBuilder &B)
    : Context(Ctx), Builder(B) {}
CIRGenTypes::~CIRGenTypes() = default;

/// ConvertType - Convert the specified type to its LLVM form.
mlir::Type CIRGenTypes::ConvertType(QualType T) {
  T = Context.getCanonicalType(T);
  const Type *Ty = T.getTypePtr();

  // For the device-side compilation, CUDA device builtin surface/texture types
  // may be represented in different types.
  assert(!Context.getLangOpts().CUDAIsDevice && "not implemented");

  // RecordTypes are cached and processed specially.
  assert(!dyn_cast<RecordType>(Ty) && "not implemented");

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
      assert("not implemented");
      break;

    case BuiltinType::Bool:
      // Note that we always return bool as i1 for use as a scalar type.
      ResultType = Builder.getI1Type();
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
      assert("not implemented");
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
      assert("not implemented");
      break;

    case BuiltinType::NullPtr:
      // Model std::nullptr_t as i8*
      // ResultType = llvm::Type::getInt8PtrTy(getLLVMContext());
      assert("not implemented");
      break;

    case BuiltinType::UInt128:
    case BuiltinType::Int128:
      assert("not implemented");
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
      assert("not implemented");
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
      assert("not implemented");
      break;
    }
#define PPC_VECTOR_TYPE(Name, Id, Size)                                        \
  case BuiltinType::Id:                                                        \
    assert("not implemented");                                                 \
    break;
#include "clang/Basic/PPCTypes.def"
#define RVV_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/RISCVVTypes.def"
      {
        assert("not implemented");
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
    assert("not implemented");
    break;
  }
  case Type::LValueReference:
  case Type::RValueReference: {
    assert("not implemented");
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
    break;
  }

  case Type::VariableArray: {
    assert("not implemented");
    break;
  }
  case Type::IncompleteArray: {
    assert("not implemented");
    break;
  }
  case Type::ConstantArray: {
    assert("not implemented");
    break;
  }
  case Type::ExtVector:
  case Type::Vector: {
    assert("not implemented");
    break;
  }
  case Type::ConstantMatrix: {
    assert("not implemented");
    break;
  }
  case Type::FunctionNoProto:
  case Type::FunctionProto:
    assert("not implemented");
    break;
  case Type::ObjCObject:
    assert("not implemented");
    break;

  case Type::ObjCInterface: {
    assert("not implemented");
    break;
  }

  case Type::ObjCObjectPointer: {
    assert("not implemented");
    break;
  }

  case Type::Enum: {
    assert("not implemented");
    break;
  }

  case Type::BlockPointer: {
    assert("not implemented");
    break;
  }

  case Type::MemberPointer: {
    assert("not implemented");
    break;
  }

  case Type::Atomic: {
    assert("not implemented");
    break;
  }
  case Type::Pipe: {
    assert("not implemented");
    break;
  }
  case Type::BitInt: {
    assert("not implemented");
    break;
  }
  }

  assert(ResultType && "Didn't convert a type?");

  TypeCache[Ty] = ResultType;
  return ResultType;
}
