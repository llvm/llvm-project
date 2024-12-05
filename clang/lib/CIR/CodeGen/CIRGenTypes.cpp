#include "CIRGenTypes.h"

#include "CIRGenModule.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"

using namespace clang;
using namespace clang::CIRGen;

CIRGenTypes::CIRGenTypes(CIRGenModule &genModule)
    : cgm(genModule), context(genModule.getASTContext()) {}

CIRGenTypes::~CIRGenTypes() {}

mlir::Type CIRGenTypes::convertType(QualType type) {
  type = context.getCanonicalType(type);
  const Type *ty = type.getTypePtr();

  mlir::Type resultType = nullptr;
  switch (ty->getTypeClass()) {
  case Type::Builtin: {
    switch (cast<BuiltinType>(ty)->getKind()) {
    // Signed types.
    case BuiltinType::Accum:
    case BuiltinType::Char_S:
    case BuiltinType::Fract:
    case BuiltinType::Int:
    case BuiltinType::Int128:
    case BuiltinType::Long:
    case BuiltinType::LongAccum:
    case BuiltinType::LongFract:
    case BuiltinType::LongLong:
    case BuiltinType::SChar:
    case BuiltinType::Short:
    case BuiltinType::ShortAccum:
    case BuiltinType::ShortFract:
    case BuiltinType::WChar_S:
      resultType = cir::IntType::get(cgm.getBuilder().getContext(),
                                     context.getTypeSize(ty),
                                     /*isSigned=*/true);
      break;
    // Unsigned types.
    case BuiltinType::Char16:
    case BuiltinType::Char32:
    case BuiltinType::Char8:
    case BuiltinType::Char_U:
    case BuiltinType::UAccum:
    case BuiltinType::UChar:
    case BuiltinType::UFract:
    case BuiltinType::UInt:
    case BuiltinType::UInt128:
    case BuiltinType::ULong:
    case BuiltinType::ULongAccum:
    case BuiltinType::ULongFract:
    case BuiltinType::ULongLong:
    case BuiltinType::UShort:
    case BuiltinType::UShortAccum:
    case BuiltinType::UShortFract:
    case BuiltinType::WChar_U:
      resultType = cir::IntType::get(cgm.getBuilder().getContext(),
                                     context.getTypeSize(ty),
                                     /*isSigned=*/false);
      break;
    default:
      cgm.errorNYI(SourceLocation(), "processing of built-in type", type);
      resultType = cir::IntType::get(cgm.getBuilder().getContext(), 32,
                                     /*isSigned=*/true);
      break;
    }
    break;
  }
  case Type::BitInt: {
    const auto *bitIntTy = cast<BitIntType>(type);
    resultType =
        cir::IntType::get(cgm.getBuilder().getContext(), bitIntTy->getNumBits(),
                          bitIntTy->isSigned());
    break;
  }
  default:
    cgm.errorNYI(SourceLocation(), "processing of type", type);
    resultType =
        cir::IntType::get(cgm.getBuilder().getContext(), 32, /*isSigned=*/true);
    break;
  }

  assert(resultType && "Type conversion not yet implemented");

  return resultType;
}
