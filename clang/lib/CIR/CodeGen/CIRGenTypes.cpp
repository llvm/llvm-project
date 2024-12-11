#include "CIRGenTypes.h"

#include "CIRGenModule.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"

using namespace clang;
using namespace clang::CIRGen;

CIRGenTypes::CIRGenTypes(CIRGenModule &genModule)
    : cgm(genModule), astContext(genModule.getASTContext()),
      builder(cgm.getBuilder()) {}

CIRGenTypes::~CIRGenTypes() {}

mlir::MLIRContext &CIRGenTypes::getMLIRContext() const {
  return *builder.getContext();
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
    default:
      cgm.errorNYI(SourceLocation(), "processing of built-in type", type);
      resultType = cgm.SInt32Ty;
      break;
    }
    break;
  }
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
