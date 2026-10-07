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

// The reader's templated readType dyn_casts and fails cleanly on mismatch.
class Ty<string type> : WithType<type, Type>;

class OptionalAttribute<string type> :
  WithParser <"succeeded($_reader.readOptionalAttribute($_var))",
  WithPrinter<"$_writer.writeOptionalAttribute($_getter)",
  WithType<type, Attribute>>>;

// For a parameter declared std::optional<...> rather than nullable: absent
// needs a flag of its own.
class OptionalInt<string type> :
  WithParser <"succeeded(readOptionalInt($_reader, $_var))",
  WithBuilder<"$_args",
  WithPrinter<"writeOptionalInt($_writer, $_getter)",
  WithType   <"std::optional<" # type # ">">>>>;

// OptionalAttribute for a parameter declared std::optional<...>. Same wire
// format, different C++ spelling.
class StdOptionalAttribute<string type> :
  WithParser <"succeeded(readStdOptionalAttribute($_reader, $_var))",
  WithBuilder<"$_args",
  WithPrinter<"writeStdOptionalAttribute($_writer, $_getter)",
  WithType   <"std::optional<" # type # ">">>>>;

// A scoped enum as a validated varint; see readEnum.
class EnumParam<string name> :
  WithParser<"succeeded(readEnum($_reader, $_var, cir::symbolize" # name # ", \"" # name # "\"))",
  WithPrinter<"$_writer.writeVarInt(static_cast<uint64_t>($_getter))",
  WithType<"cir::" # name>>>;

def CtorKindParam : EnumParam<"CtorKind">;
def AssignKindParam : EnumParam<"AssignKind">;

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

def OptInfoAttr : DialectAttribute<(attr
  CheckedUnsigned:$level,
  CheckedUnsigned:$size
)>;

// Negative offsets are legitimate.
def AddressPointAttr : DialectAttribute<(attr
  CheckedInt32:$index,
  CheckedInt32:$offset
)>;

def OpenCLVersionAttr : DialectAttribute<(attr
  CheckedInt32:$major,
  CheckedInt32:$minor
)>;

def BlockAddrInfoAttr : DialectAttribute<(attr
  Ty<"cir::PointerType">:$type,
  Attr<"mlir::FlatSymbolRefAttr">:$func,
  Attr<"mlir::StringAttr">:$label
)>;

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
// Every literal in a function body is one of these, so each fallback
// occurrence costs its full printed form.

// The payload width is implied by the type and not stored twice, mirroring
// builtin's IntegerAttr / FloatAttr.
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

// BoolType is a singleton, so the type is rebuilt on read rather than stored.
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
// No payload: each costs a kind index instead of its mnemonic.

def CatchAllAttr : DialectAttribute<(attr)>;
def UnwindAttr : DialectAttribute<(attr)>;

//===----------------------------------------------------------------------===//
// Constant initializers
//===----------------------------------------------------------------------===//
//
// Each occurrence embeds its self type's full printed form; encoded, it
// writes a type-table reference instead.

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
)>;

def ConstRecordAttr : DialectAttribute<(attr
  Type:$type,
  Attr<"mlir::ArrayAttr">:$members
)>;

// Each carries the aggregate type, which the fallback prints fully expanded.
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

// Containers only pay off once their nested attributes are encoded too.
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
)>;

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
