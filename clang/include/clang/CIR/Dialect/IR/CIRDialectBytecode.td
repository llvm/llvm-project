//===-- CIRDialectBytecode.td - CIR bytecode defs ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the CIR bytecode reader/writer definition file.
//
// Without a BytecodeDialectInterface, MLIR falls back to encoding an unknown
// dialect's attributes and types through their *assembly format*: it embeds
// the printed text inside the bytecode. That fallback is correct but it is the
// reason a bytecode file can be larger than the equivalent text.
//
// Modelled on mlir/include/mlir/Dialect/LLVMIR/LLVMDialectBytecode.td. Every
// entry's parameters must appear in the same order as the attribute's get(),
// because the generated reader feeds them positionally, and each name must
// derive the C++ getter (`get` + the name in UpperCamel).
//
//===----------------------------------------------------------------------===//

#ifndef CIR_DIALECT_BYTECODE
#define CIR_DIALECT_BYTECODE

include "mlir/IR/BytecodeBase.td"

//===----------------------------------------------------------------------===//
// Bytecode classes for attributes and types.
//===----------------------------------------------------------------------===//

class Attr<string type> : WithType<type, Attribute>;

// Read a generic type into a narrowed C++ handle. The reader's templated
// readType does the dyn_cast and fails cleanly on a mismatch, so this stays
// safe against a malformed or cross-version file.
class Ty<string type> : WithType<type, Type>;

class OptionalAttribute<string type> :
  WithParser <"succeeded($_reader.readOptionalAttribute($_var))",
  WithPrinter<"$_writer.writeOptionalAttribute($_getter)",
  WithType<type, Attribute>>>;

// For a parameter declared OptionalParameter<"std::optional<...>">, where the
// C++ type is an optional rather than a nullable attribute, so "absent" needs
// a flag of its own. The helpers live in CIRDialectBytecode.cpp.
class OptionalInt<string type> :
  WithParser <"succeeded(readOptionalInt($_reader, $_var))",
  WithBuilder<"$_args",
  WithPrinter<"writeOptionalInt($_writer, $_getter)",
  WithType   <"std::optional<" # type # ">">>>>;

// Named for the C++ spelling, not the concept, because the difference from
// OptionalAttribute above is exactly the C++ spelling and nothing else:
// OptionalAttribute is for a parameter that is a nullable attribute, this is
// for one declared std::optional<...>. The two share the same wire format, so
// at a glance nothing but the name says which a given parameter needs.
class StdOptionalAttribute<string type> :
  WithParser <"succeeded(readStdOptionalAttribute($_reader, $_var))",
  WithBuilder<"$_args",
  WithPrinter<"writeStdOptionalAttribute($_writer, $_getter)",
  WithType   <"std::optional<" # type # ">">>>>;

// A scoped enum read from an untrusted varint. The value is validated with
// the enum's symbolizer; an out-of-range value fails the read with a
// diagnostic instead of casting into undefined behavior.
def CtorKindParam :
  WithParser<"succeeded(readEnum($_reader, $_var, cir::symbolizeCtorKind, \"CtorKind\"))",
  WithPrinter<"$_writer.writeVarInt(static_cast<uint64_t>($_getter))",
  WithType<"cir::CtorKind">>>;

def AssignKindParam :
  WithParser<"succeeded(readEnum($_reader, $_var, cir::symbolizeAssignKind, \"AssignKind\"))",
  WithPrinter<"$_writer.writeVarInt(static_cast<uint64_t>($_getter))",
  WithType<"cir::AssignKind">>>;

// A varint on the wire narrowed to a fixed-width integer. The bound is checked
// on read; a file carrying a value the parameter cannot hold fails instead of
// truncating into a plausible count or index.
def CheckedInt :
  WithParser<"succeeded(readCheckedInt($_reader, $_var))",
  WithPrinter<"$_writer.writeSignedVarInt($_getter)",
  WithType<"int">>>;

def CheckedInt32 :
  WithParser<"succeeded(readCheckedInt($_reader, $_var))",
  WithPrinter<"$_writer.writeSignedVarInt($_getter)",
  WithType<"int32_t">>>;

def CheckedUnsigned :
  WithParser<"succeeded(readCheckedInt($_reader, $_var))",
  WithPrinter<"$_writer.writeVarInt($_getter)",
  WithType<"unsigned">>>;

//===----------------------------------------------------------------------===//
// OptInfoAttr
//===----------------------------------------------------------------------===//

def OptInfoAttr : DialectAttribute<(attr
  CheckedUnsigned:$level,
  CheckedUnsigned:$size
)>;

//===----------------------------------------------------------------------===//
// AddressPointAttr
//
// index and offset are int32_t and an address point legitimately carries a
// negative offset, so these are signed.
//===----------------------------------------------------------------------===//

def AddressPointAttr : DialectAttribute<(attr
  CheckedInt32:$index,
  CheckedInt32:$offset
)>;

//===----------------------------------------------------------------------===//
// OpenCLVersionAttr
//===----------------------------------------------------------------------===//

def OpenCLVersionAttr : DialectAttribute<(attr
  CheckedInt32:$major,
  CheckedInt32:$minor
)>;

//===----------------------------------------------------------------------===//
// BlockAddrInfoAttr
//===----------------------------------------------------------------------===//

def BlockAddrInfoAttr : DialectAttribute<(attr
  Ty<"cir::PointerType">:$type,
  Attr<"mlir::FlatSymbolRefAttr">:$func,
  Attr<"mlir::StringAttr">:$label
)>;

//===----------------------------------------------------------------------===//
// BitfieldInfoAttr
//===----------------------------------------------------------------------===//

def BitfieldInfoAttr : DialectAttribute<(attr
  Attr<"mlir::StringAttr">:$name,
  Type:$storage_type,
  VarInt:$size,
  VarInt:$offset,
  Bool:$is_signed
)>;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//
//
// These are the entries that pay for themselves. Every literal in a function
// body is an IntAttr, an FPAttr or a BoolAttr, so under the assembly fallback
// each one costs its full printed form ("#cir.int<42> : !s32i") while the
// native encoding costs a type reference plus the payload's own width.

// The value's width is implied by the type, so it is not stored twice; the
// helpers below recover it. This mirrors builtin's IntegerAttr / FloatAttr.
def IntAttr : DialectAttribute<(attr
  Type:$type,
  KnownWidthAPInt<"type">:$value
)> {
  let cBuilder = "get<$_resultType>(context, mlir::cast<cir::IntTypeInterface>(type), *value)";
}

def FPAttr : DialectAttribute<(attr
  Type:$type,
  KnownSemanticsAPFloat<"type">:$value
)> {
  let cBuilder = "get<$_resultType>(context, mlir::cast<cir::FPTypeInterface>(type), *value)";
}

// cir::BoolType is a singleton, so the self type carries no information and is
// rebuilt on read rather than stored.
def BoolAttr : DialectAttribute<(attr
  Bool:$value
)> {
  let cBuilder = "get<$_resultType>(context, cir::BoolType::get(context), value)";
}

def ConstPtrAttr : DialectAttribute<(attr
  Ty<"cir::PointerType">:$type,
  Attr<"mlir::IntegerAttr">:$value
)>;

//===----------------------------------------------------------------------===//
// Parameterless marker attributes
//===----------------------------------------------------------------------===//
//
// No payload at all: the entry exists so the attribute costs a kind index
// instead of its printed mnemonic.

def CatchAllAttr : DialectAttribute<(attr)>;
def UnwindAttr : DialectAttribute<(attr)>;

//===----------------------------------------------------------------------===//
// Constant initializers
//===----------------------------------------------------------------------===//
//
// These carry a self type, and under the fallback each *occurrence* embeds
// that type's whole printed form, so a zero-initialized aggregate costs the
// full type text at every use. Encoding them natively writes a reference
// into the bytecode type table instead.

def ZeroAttr : DialectAttribute<(attr
  Type:$type
)>;

def UndefAttr : DialectAttribute<(attr
  Type:$type
)>;

def PoisonAttr : DialectAttribute<(attr
  Type:$type
)>;

def ConstArrayAttr : DialectAttribute<(attr
  Type:$type,
  Attribute:$elts,
  CheckedInt:$trailingZerosNum
)> {
  let cBuilder = "getChecked<$_resultType>([&]() { return reader.emitError(); }, context, type, elts, trailingZerosNum)";
}

def ConstRecordAttr : DialectAttribute<(attr
  Type:$type,
  Attr<"mlir::ArrayAttr">:$members
)>;

// The C++ special-member attributes on cir.func. Each is small, a type and a
// flag, but its type is the aggregate the member belongs to, and while these
// fell back the printed form dragged that type in fully expanded.
def CXXCtorAttr : DialectAttribute<(attr
  Type:$type,
  CtorKindParam:$ctorKind,
  Bool:$isTrivial
)>;

def CXXDtorAttr : DialectAttribute<(attr
  Type:$type,
  Bool:$isTrivial
)>;

def CXXAssignAttr : DialectAttribute<(attr
  Type:$type,
  AssignKindParam:$assignKind,
  Bool:$isTrivial
)>;

def VTableAttr : DialectAttribute<(attr
  Type:$type,
  Attr<"mlir::ArrayAttr">:$data
)>;

def BlockAddrDiffAttr : DialectAttribute<(attr
  Ty<"cir::IntType">:$type,
  Attr<"mlir::FlatSymbolRefAttr">:$func,
  Attr<"mlir::StringAttr">:$lhs_label,
  Attr<"mlir::StringAttr">:$rhs_label
)>;

// The rest of the type-carrying attributes. A container that nests other
// attributes only pays off once those are encoded too: the container would
// otherwise write its operands as nested attributes, each of which drops
// straight back to the fallback and prints its own self type in full.
def GlobalViewAttr : DialectAttribute<(attr
  Type:$type,
  Attr<"mlir::FlatSymbolRefAttr">:$symbol,
  OptionalAttribute<"mlir::ArrayAttr">:$indices
)>;

def TypeInfoAttr : DialectAttribute<(attr
  Type:$type,
  Attr<"mlir::ArrayAttr">:$data
)>;

def DynamicCastInfoAttr : DialectAttribute<(attr
  Attr<"cir::GlobalViewAttr">:$src_rtti,
  Attr<"cir::GlobalViewAttr">:$dest_rtti,
  Attr<"mlir::FlatSymbolRefAttr">:$runtime_func,
  Attr<"mlir::FlatSymbolRefAttr">:$bad_cast_func,
  Attr<"cir::IntAttr">:$offset_hint
)>;

def ConstVectorAttr : DialectAttribute<(attr
  Type:$type,
  Attr<"mlir::ArrayAttr">:$elts
)>;

def ConstComplexAttr : DialectAttribute<(attr
  Ty<"cir::ComplexType">:$type,
  Attr<"mlir::TypedAttr">:$real,
  Attr<"mlir::TypedAttr">:$imag
)>;

def DataMemberAttr : DialectAttribute<(attr
  Ty<"cir::DataMemberType">:$type,
  OptionalAttribute<"mlir::DenseI32ArrayAttr">:$member_path
)> {
  // Not the default builder: that one resolves to the ArrayRef<int32_t>
  // overload through DenseI32ArrayAttr's implicit conversion, which
  // dereferences a null attribute for a null pointer-to-member. The context
  // form takes the attribute as-is.
  let cBuilder = "cir::DataMemberAttr::getChecked([&]() { return reader.emitError(); }, context, type, member_path)";
}

def MethodAttr : DialectAttribute<(attr
  Ty<"cir::MethodType">:$type,
  StdOptionalAttribute<"mlir::FlatSymbolRefAttr">:$symbol,
  OptionalInt<"uint64_t">:$vtable_offset
)>;

def CIRDialectAttributes : DialectAttributes<"CIR"> {
  let elems = [
    OptInfoAttr,
    AddressPointAttr,
    OpenCLVersionAttr,
    BlockAddrInfoAttr,
    BitfieldInfoAttr,
    IntAttr,
    FPAttr,
    BoolAttr,
    ConstPtrAttr,
    CatchAllAttr,
    UnwindAttr,
    ZeroAttr,
    UndefAttr,
    PoisonAttr,
    ConstArrayAttr,
    ConstRecordAttr,
    VTableAttr,
    BlockAddrDiffAttr,
    CXXCtorAttr,
    CXXDtorAttr,
    CXXAssignAttr,
    GlobalViewAttr,
    TypeInfoAttr,
    DynamicCastInfoAttr,
    ConstVectorAttr,
    ConstComplexAttr,
    DataMemberAttr,
    MethodAttr,
    // APPEND ONLY. An entry's position in this list *is* its kind
    // discriminator on disk, so inserting one renumbers every entry after
    // it. No dialect version is recorded yet, so this is a same-revision
    // scheme, not a compatibility promise.
  ];
}

#endif // CIR_DIALECT_BYTECODE
