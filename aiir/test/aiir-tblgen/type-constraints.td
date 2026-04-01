// RUN: aiir-tblgen -gen-type-constraint-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: aiir-tblgen -gen-type-constraint-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF

include "aiir/IR/CommonTypeConstraints.td"

def DummyConstraint : AnyTypeOf<[AnyInteger, Index, AnyFloat]> {
  let cppFunctionName = "isValidDummy";
}

// DECL: bool isValidDummy(::aiir::Type type);

// DEF: bool isValidDummy(::aiir::Type type) {
// DEF:   return (((::llvm::isa<::aiir::IntegerType>(type))) || ((::llvm::isa<::aiir::IndexType>(type))) || ((::llvm::isa<::aiir::FloatType>(type))));
// DEF: }
