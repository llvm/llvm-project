//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_IR_CIRATTRS_TD
#define LLVM_CLANG_CIR_DIALECT_IR_CIRATTRS_TD

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

include "clang/CIR/Dialect/IR/CIRDialect.td"

//===----------------------------------------------------------------------===//
// CIR Attrs
//===----------------------------------------------------------------------===//

class CIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class CIRUnitAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : CIR_Attr<name, attrMnemonic, traits> {
  let returnType = "bool";
  let defaultValue = "false";
  let valueType = NoneType;
  let isOptional = 1;
}

//===----------------------------------------------------------------------===//
// IntegerAttr
//===----------------------------------------------------------------------===//

def IntAttr : CIR_Attr<"Int", "int", [TypedAttrInterface]> {
  let summary = "An attribute containing an integer value";
  let description = [{
    An integer attribute is a literal attribute that represents an integral
    value of the specified integer type.
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "llvm::APInt":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type,
                                        "const llvm::APInt &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>,
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type,
                                        "int64_t":$value), [{
      IntType intType = mlir::cast<IntType>(type);
      mlir::APInt apValue(intType.getWidth(), value, intType.isSigned());
      return $_get(intType.getContext(), intType, apValue);
    }]>,
  ];
  let extraClassDeclaration = [{
    int64_t getSInt() const { return getValue().getSExtValue(); }
    uint64_t getUInt() const { return getValue().getZExtValue(); }
    bool isNullValue() const { return getValue() == 0; }
    uint64_t getBitWidth() const {
      return mlir::cast<IntType>(getType()).getWidth();
    }
  }];
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// FPAttr
//===----------------------------------------------------------------------===//

def FPAttr : CIR_Attr<"FP", "fp", [TypedAttrInterface]> {
  let summary = "An attribute containing a floating-point value";
  let description = [{
    An fp attribute is a literal attribute that represents a floating-point
    value of the specified floating-point type. Supporting only CIR FP types.
  }];
  let parameters = (ins
    AttributeSelfTypeParameter<"", "::cir::CIRFPTypeInterface">:$type,
    APFloatParameter<"">:$value
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type,
                                        "const llvm::APFloat &":$value), [{
      return $_get(type.getContext(), mlir::cast<CIRFPTypeInterface>(type),
                   value);
    }]>,
    AttrBuilder<(ins "mlir::Type":$type,
                     "const llvm::APFloat &":$value), [{
      return $_get($_ctxt, mlir::cast<CIRFPTypeInterface>(type), value);
    }]>,
  ];
  let extraClassDeclaration = [{
    static FPAttr getZero(mlir::Type type);
  }];
  let genVerifyDecl = 1;

  let assemblyFormat = [{
    `<` custom<FloatLiteral>($value, ref($type)) `>`
  }];
}

//===----------------------------------------------------------------------===//
// ConstPtrAttr
//===----------------------------------------------------------------------===//

def ConstPtrAttr : CIR_Attr<"ConstPtr", "ptr", [TypedAttrInterface]> {
  let summary = "Holds a constant pointer value";
  let parameters = (ins
    AttributeSelfTypeParameter<"", "::cir::PointerType">:$type,
    "mlir::IntegerAttr":$value);
  let description = [{
    A pointer attribute is a literal attribute that represents an integral
    value of a pointer type.
  }];
  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type,
                                        "mlir::IntegerAttr":$value), [{
      return $_get(type.getContext(), mlir::cast<cir::PointerType>(type),
                   value);
    }]>,
    AttrBuilder<(ins "mlir::Type":$type,
                     "mlir::IntegerAttr":$value), [{
      return $_get($_ctxt, mlir::cast<cir::PointerType>(type), value);
    }]>,
  ];
  let extraClassDeclaration = [{
    bool isNullValue() const { return getValue().getInt() == 0; }
  }];

  let assemblyFormat = [{
    `<` custom<ConstPtr>($value) `>`
  }];
}

#endif // LLVM_CLANG_CIR_DIALECT_IR_CIRATTRS_TD
