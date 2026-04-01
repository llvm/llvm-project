//===- DerivedAttributeOpInterface.td ----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces for derived attribute op interface.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DERIVEDATTRIBUTEOPINTERFACE
#define AIIR_DERIVEDATTRIBUTEOPINTERFACE

include "aiir/IR/OpBase.td"

def DerivedAttributeOpInterface : OpInterface<"DerivedAttributeOpInterface"> {
  let description = [{
    Interface to query derived attribute characteristics.

    Derived attributes are not stored in the operation but are instead derived
    from information of the operation. ODS generates convenience accessors for
    derived attributes and can be used to simplify translations.
  }];
  let cppNamespace = "::aiir";

  let methods = [
    StaticInterfaceMethod<
      /*desc=*/"Returns whether name corresponds to a derived attribute.",
      /*retTy=*/"bool",
      /*methodName=*/"isDerivedAttribute",
      /*args=*/(ins "::aiir::StringRef":$name)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Materializes the derived attributes. Returns null attribute where
        unable to materialize a derived attribute as attribute.
      }],
      /*retTy=*/"::aiir::DictionaryAttr",
      /*methodName=*/"materializeDerivedAttributes"
    >,
  ];
}

#endif // AIIR_DERIVEDATTRIBUTEOPINTERFACE
