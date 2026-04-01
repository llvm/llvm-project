// RUN: aiir-tblgen -gen-attr-constraint-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: aiir-tblgen -gen-attr-constraint-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF

include "aiir/IR/CommonAttrConstraints.td"

def DummyConstraint : AnyAttrOf<[APIntAttr, ArrayAttr, UnitAttr]> {
  let cppFunctionName = "isValidDummy";
}

// DECL: bool isValidDummy(::aiir::Attribute attr);

// DEF: bool isValidDummy(::aiir::Attribute attr) {
// DEF:   return (((::llvm::isa<::aiir::IntegerAttr>(attr))) || ((::llvm::isa<::aiir::ArrayAttr>(attr))) || ((::llvm::isa<::aiir::UnitAttr>(attr))));
// DEF: }
