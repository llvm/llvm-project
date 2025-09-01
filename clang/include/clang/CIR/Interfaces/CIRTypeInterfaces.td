//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines cir type interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_INTERFACES_CIRTYPEINTERFACES_TD
#define CLANG_CIR_INTERFACES_CIRTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

def CIR_IntTypeInterface : TypeInterface<"IntTypeInterface"> {
  let description = [{
    Contains helper functions to query properties about an integer type.
  }];
  let cppNamespace = "::cir";
  let methods = [
    InterfaceMethod<[{
        Returns true if this is a signed integer type.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isSigned",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_type.isSigned();
      }]
    >,
    InterfaceMethod<[{
        Returns true if this is an unsigned integer type.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isUnsigned",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_type.isUnsigned();
      }]
    >,
    InterfaceMethod<[{
        Returns the bit width of this integer type.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getWidth",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_type.getWidth();
      }]
    >
  ];
}

def CIR_FPTypeInterface : TypeInterface<"FPTypeInterface"> {
  let description = [{
    Contains helper functions to query properties about a floating-point type.
  }];
  let cppNamespace = "::cir";

  let methods = [
    InterfaceMethod<[{
        Returns the bit width of this floating-point type.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getWidth",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
          return llvm::APFloat::semanticsSizeInBits($_type.getFloatSemantics());
        }]
    >,
    InterfaceMethod<[{
        Return the mantissa width.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getFPMantissaWidth",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
          return llvm::APFloat::semanticsPrecision($_type.getFloatSemantics());
        }]
    >,
    InterfaceMethod<[{
        Return the float semantics of this floating-point type.
      }],
      /*retTy=*/"const llvm::fltSemantics &",
      /*methodName=*/"getFloatSemantics"
    >,
  ];
}

def CIR_SizedTypeInterface : TypeInterface<"SizedTypeInterface"> {
  let description = [{
    Annotates types that have known size. Types that don't have a size are
    abstract types and void.
  }];
  let cppNamespace = "::cir";
  let methods = [
    InterfaceMethod<[{
        Returns true if this is a sized type. This mirrors sizedness from the
        clang AST, where a type is sized if it has a known size.
        By default type defining this interface returns true,
        but this can be overridden if sizedness depends on properties of the type.
        For example, whether a struct is not sized if it is incomplete.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isSized",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return true;
      }]
    >,
  ];
}

#endif // CLANG_CIR_INTERFACES_CIRTYPEINTERFACES_TD
