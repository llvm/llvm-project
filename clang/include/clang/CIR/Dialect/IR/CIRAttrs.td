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

class CIR_TypedAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : CIR_Attr<name, attrMnemonic, !listconcat(traits, [TypedAttrInterface])> {

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type), [{
      return $_get(type.getContext(), type);
    }]>
  ];

  let assemblyFormat = [{}];
}

class CIRUnitAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : CIR_Attr<name, attrMnemonic, traits> {
  let returnType = "bool";
  let defaultValue = "false";
  let valueType = NoneType;
  let isOptional = 1;
}

//===----------------------------------------------------------------------===//
// BoolAttr
//===----------------------------------------------------------------------===//

def CIR_BoolAttr : CIR_Attr<"Bool", "bool", [TypedAttrInterface]> {
  let summary = "Represent true/false for !cir.bool types";
  let description = [{
    The BoolAttr represents a 'true' or 'false' value.
  }];

  let parameters = (ins AttributeSelfTypeParameter<
                        "", "cir::BoolType">:$type,
                    "bool":$value);

  let builders = [
    AttrBuilder<(ins "bool":$value), [{
      return $_get($_ctxt, cir::BoolType::get($_ctxt), value);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $value `>`
  }];
}

//===----------------------------------------------------------------------===//
// ZeroAttr
//===----------------------------------------------------------------------===//

def ZeroAttr : CIR_TypedAttr<"Zero", "zero"> {
  let summary = "Attribute to represent zero initialization";
  let description = [{
    The ZeroAttr is used to indicate zero initialization on structs.
  }];
}

//===----------------------------------------------------------------------===//
// UndefAttr
//===----------------------------------------------------------------------===//

def UndefAttr : CIR_TypedAttr<"Undef", "undef"> {
  let summary = "Represent an undef constant";
  let description = [{
    The UndefAttr represents an undef constant, corresponding to LLVM's notion
    of undef.
  }];
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
                        APIntParameter<"">:$value);
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
// ConstArrayAttr
//===----------------------------------------------------------------------===//

def ConstArrayAttr : CIR_Attr<"ConstArray", "const_array", [TypedAttrInterface]> {
  let summary = "A constant array from ArrayAttr or StringRefAttr";
  let description = [{
    An CIR array attribute is an array of literals of the specified attr types.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "mlir::Attribute":$elts,
                        "int":$trailingZerosNum);

  // Define a custom builder for the type; that removes the need to pass
  // in an MLIRContext instance, as it can be infered from the `type`.
  let builders = [
    AttrBuilderWithInferredContext<(ins "cir::ArrayType":$type,
                                        "mlir::Attribute":$elts), [{
      int zeros = 0;
      auto typeSize = mlir::cast<cir::ArrayType>(type).getSize();
      if (auto str = mlir::dyn_cast<mlir::StringAttr>(elts))
        zeros = typeSize - str.size();
      else
        zeros = typeSize - mlir::cast<mlir::ArrayAttr>(elts).size();

      return $_get(type.getContext(), type, elts, zeros);
    }]>,
    AttrBuilderWithInferredContext<(ins "cir::ArrayType":$type,
                                        "mlir::Attribute":$elts,
                                        "int":$trailingZerosNum), [{
      return $_get(type.getContext(), type, elts, trailingZerosNum);
    }]>
  ];

  // Printing and parsing available in CIRAttrs.cpp
  let hasCustomAssemblyFormat = 1;

  // Enable verifier.
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    bool hasTrailingZeros() const { return getTrailingZerosNum() != 0; };
  }];
}

//===----------------------------------------------------------------------===//
// ConstVectorAttr
//===----------------------------------------------------------------------===//

def ConstVectorAttr : CIR_Attr<"ConstVector", "const_vector",
                               [TypedAttrInterface]> {
  let summary = "A constant vector from ArrayAttr";
  let description = [{
    A CIR vector attribute is an array of literals of the specified attribute
    types.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
		        "mlir::ArrayAttr":$elts);

  // Define a custom builder for the type; that removes the need to pass in an
  // MLIRContext instance, as it can be inferred from the `type`.
  let builders = [
    AttrBuilderWithInferredContext<(ins "cir::VectorType":$type,
		                        "mlir::ArrayAttr":$elts), [{
      return $_get(type.getContext(), type, elts);
    }]>
  ];

  let assemblyFormat = [{
    `<` $elts `>`
  }];

  // Enable verifier.
  let genVerifyDecl = 1;
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
    }]>
  ];
  let extraClassDeclaration = [{
    bool isNullValue() const { return getValue().getInt() == 0; }
  }];

  let assemblyFormat = [{
    `<` custom<ConstPtr>($value) `>`
  }];
}

#endif // LLVM_CLANG_CIR_DIALECT_IR_CIRATTRS_TD
