//===- CIRTypes.td - CIR dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_DIALECT_CIR_TYPES
#define MLIR_CIR_DIALECT_CIR_TYPES

include "clang/CIR/Dialect/IR/CIRDialect.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// CIR Types
//===----------------------------------------------------------------------===//

class CIR_Type<string name, string typeMnemonic, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<CIR_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// IntType
//===----------------------------------------------------------------------===//

def CIR_IntType : CIR_Type<"Int", "int",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Integer type with arbitrary precision up to a fixed limit";
  let description = [{
    CIR type that represents integer types with arbitrary precision, including
    standard integral types such as `int` and `long`, extended integral types
    such as `__int128`, and arbitrary width types such as `_BitInt(n)`.

    Those integer types that are directly available in C/C++ standard are called
    primitive integer types. Said types are: `signed char`, `short`, `int`,
    `long`, `long long`, and their unsigned variations.
  }];
  let parameters = (ins "unsigned":$width, "bool":$isSigned);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    /// Return true if this is a signed integer type.
    bool isSigned() const { return getIsSigned(); }
    /// Return true if this is an unsigned integer type.
    bool isUnsigned() const { return !getIsSigned(); }
    /// Return type alias.
    std::string getAlias() const {
      return (isSigned() ? 's' : 'u') + std::to_string(getWidth()) + 'i';
    }
    /// Return true if this is a primitive integer type (i.e. signed or unsigned
    /// integer types whose bit width is 8, 16, 32, or 64).
    bool isPrimitive() const {
      return isValidPrimitiveIntBitwidth(getWidth());
    }
    bool isSignedPrimitive() const {
      return isPrimitive() && isSigned();
    }

    /// Returns a minimum bitwidth of cir::IntType
    static unsigned minBitwidth() { return 1; }
    /// Returns a maximum bitwidth of cir::IntType
    static unsigned maxBitwidth() { return 128; }

    /// Returns true if cir::IntType that represents a primitive integer type
    /// can be constructed from the provided bitwidth.
    static bool isValidPrimitiveIntBitwidth(unsigned width) {
      return width == 8 || width == 16 || width == 32 || width == 64;
    }
  }];
  let genVerifyDecl = 1;
}

// Constraints

// Unsigned integer type of a specific width.
class UInt<int width>
  : Type<And<[
	CPred<"::mlir::isa<::cir::IntType>($_self)">,
	CPred<"::mlir::cast<::cir::IntType>($_self).isUnsigned()">,
	CPred<"::mlir::cast<::cir::IntType>($_self).getWidth() == " # width>
        ]>, width # "-bit unsigned integer", "::cir::IntType">,
    BuildableType<
      "cir::IntType::get($_builder.getContext(), "
      # width # ", /*isSigned=*/false)"> {
  int bitwidth = width;
}

def UInt1  : UInt<1>;
def UInt8  : UInt<8>;
def UInt16 : UInt<16>;
def UInt32 : UInt<32>;
def UInt64 : UInt<64>;

// Signed integer type of a specific width.
class SInt<int width>
  : Type<And<[
	CPred<"::mlir::isa<::cir::IntType>($_self)">,
	CPred<"::mlir::cast<::cir::IntType>($_self).isSigned()">,
	CPred<"::mlir::cast<::cir::IntType>($_self).getWidth() == " # width>
        ]>, width # "-bit signed integer", "::cir::IntType">,
    BuildableType<
      "cir::IntType::get($_builder.getContext(), "
      # width # ", /*isSigned=*/true)"> {
  int bitwidth = width;
}

def SInt1  : SInt<1>;
def SInt8  : SInt<8>;
def SInt16 : SInt<16>;
def SInt32 : SInt<32>;
def SInt64 : SInt<64>;

def PrimitiveUInt
    : AnyTypeOf<[UInt8, UInt16, UInt32, UInt64], "primitive unsigned int",
                "::cir::IntType">;

def PrimitiveSInt
    : AnyTypeOf<[SInt8, SInt16, SInt32, SInt64], "primitive signed int",
                "::cir::IntType">;

def PrimitiveInt
    : AnyTypeOf<[UInt8, UInt16, UInt32, UInt64, SInt8, SInt16, SInt32, SInt64],
                "primitive int", "::cir::IntType">;

#endif // MLIR_CIR_DIALECT_CIR_TYPES
