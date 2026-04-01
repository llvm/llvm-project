//===- ComplexAttributes.td - Definitions for complex attributes -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMPLEX_ATTRIBUTE
#define COMPLEX_ATTRIBUTE

include "aiir/IR/AttrTypeBase.td"
include "aiir/IR/BuiltinAttributeInterfaces.td"
include "aiir/Dialect/Complex/IR/ComplexBase.td"

//===----------------------------------------------------------------------===//
// Complex Attributes.
//===----------------------------------------------------------------------===//

class Complex_Attr<string attrName, string attrMnemonic,
                   list<Trait> traits = []>
    : AttrDef<Complex_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def Complex_NumberAttr : Complex_Attr<"Number", "number",
                                      [TypedAttrInterface]> {
  let summary = "A complex number attribute";

  let description = [{
    A complex number attribute.

    Example:

    ```aiir
    #complex.number<:f64 1.0, 2.0>
    ```
  }];

  let parameters = (ins APFloatParameter<"">:$real,
                        APFloatParameter<"">:$imag,
                        AttributeSelfTypeParameter<"">:$type);
  let builders = [
    AttrBuilderWithInferredContext<(ins "aiir::ComplexType":$type,
                                        "double":$real,
                                        "double":$imag), [{
      auto elementType = ::llvm::cast<FloatType>(type.getElementType());
      APFloat realFloat(real);
      bool unused;
      realFloat.convert(elementType.getFloatSemantics(),
                        APFloat::rmNearestTiesToEven, &unused);
      APFloat imagFloat(imag);
      imagFloat.convert(elementType.getFloatSemantics(),
                        APFloat::rmNearestTiesToEven, &unused);
      return $_get(type.getContext(), realFloat, imagFloat, type);
    }]>
  ];

  let extraClassDeclaration = [{
    std::complex<APFloat> getValue() {
      return std::complex<APFloat>(getReal(), getImag());
    }
  }];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

#endif // COMPLEX_ATTRIBUTE
