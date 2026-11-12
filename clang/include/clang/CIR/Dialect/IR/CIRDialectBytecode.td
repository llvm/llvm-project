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
// Modelled on mlir/include/mlir/Dialect/LLVMIR/LLVMDialectBytecode.td.
// With the default cBuilder, parameters must match the builder's argument
// order. Parameter names determine the generated getter names.
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

// A type parameter that may be null; presence is a flag of its own.
def OptionalType :
  WithParser <"succeeded(readOptionalType($_reader, $_var))",
  WithPrinter<"writeOptionalType($_writer, $_getter)",
  WithType<"mlir::Type", Type>>>;

def ElementCountParam :
  WithParser <"succeeded(readElementCount($_reader, $_var))",
  WithBuilder<"$_args",
  WithPrinter<"writeElementCount($_writer, $_getter)",
  WithType   <"llvm::ElementCount">>>>;

// A whole record, read and written by readRecord and writeRecord.
class RecordParam<string type> :
  WithParser <"succeeded(readRecord($_reader, $_var))",
  WithBuilder<"$_args",
  WithPrinter<"writeRecord($_writer, $_getter)",
  WithGetter <"$_attrType",
  WithType   <type>>>>>;

// The default builder only finds a getChecked without a context and otherwise
// builds unchecked. A type whose verifying getChecked takes one uses this;
// `args` converts what the default arguments do not.
class CheckedDialectType<dag d, string args = "$_args"> : DialectType<d> {
  let cBuilder = "$_resultType::getChecked([&]() { return reader.emitError(); }, context, " # args # ")";
}

// A struct or union, named by the def.
class RecordDialectType :
    DialectType<(type RecordParam<"cir::" # NAME>:$record)> {
  let cBuilder = "$_args";
}

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
)> {
  let cBuilder = "$_resultType::getChecked([&]() { return reader.emitError(); }, context, $_args)";
}

// Negative offsets are legitimate.
def AddressPointAttr : DialectAttribute<(attr
  CheckedInt32:$index,
  CheckedInt32:$offset
)>;

def OpenCLVersionAttr : DialectAttribute<(attr
  CheckedInt32:$majorVersion,
  CheckedInt32:$minorVersion
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
)> {
  let cBuilder = "$_resultType::getChecked([&]() { return reader.emitError(); }, context, $_args)";
}

def ConstRecordAttr : DialectAttribute<(attr
  Type:$type,
  Attr<"mlir::ArrayAttr">:$members
)> {
  let cBuilder = "$_resultType::getChecked([&]() { return reader.emitError(); }, context, $_args)";
}

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
  OptionalAttribute<"mlir::ArrayAttr">:$indices,
  Bool:$address_point
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
)> {
  let cBuilder = "$_resultType::getChecked([&]() { return reader.emitError(); }, context, $_args)";
}

def BitFieldDeclAttr : DialectAttribute<(attr
  Type:$declared_type,
  VarInt:$width,
  Bool:$is_unnamed
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
    BitFieldDeclAttr,
    // APPEND ONLY. An entry's position in this list *is* its kind
    // discriminator on disk, so inserting one renumbers every entry after
    // it. No dialect version is recorded yet, so this is a same-revision
    // scheme, not a compatibility promise.
  ];
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//
//
// Without an encoding, a type is stored as its printed form, which expands
// every record it reaches through a pointer. Every type that refers to one
// stores a type-table reference instead, so each record body is stored once.

def IntType : CheckedDialectType<(type
  CheckedUnsigned:$width,
  Bool:$is_signed,
  Bool:$is_bit_int
)>;

def SingleType : DialectType<(type)>;
def DoubleType : DialectType<(type)>;
def FP16Type : DialectType<(type)>;
def BF16Type : DialectType<(type)>;
def FP80Type : DialectType<(type)>;
def FP128Type : DialectType<(type)>;

def LongDoubleType : CheckedDialectType<(type
  Type:$underlying
)>;

def ComplexType : DialectType<(type
  Type:$elementType
)>;

def PointerType : DialectType<(type
  Type:$pointee,
  OptionalAttribute<"mlir::ptr::MemorySpaceAttrInterface">:$addrSpace
)>;

def DataMemberType : DialectType<(type
  Type:$member_ty,
  Ty<"cir::RecordType">:$class_ty
)>;

def VPtrType : DialectType<(type)>;
def CUDADeviceSurfaceType : DialectType<(type)>;
def CUDADeviceTextureType : DialectType<(type)>;
def BoolType : DialectType<(type)>;

def ArrayType : DialectType<(type
  Type:$elementType,
  VarInt:$size
)>;

def VectorType : DialectType<(type
  Type:$element_type,
  ElementCountParam:$size
)>;

def MatrixType : DialectType<(type
  Type:$element_type,
  VarInt:$num_rows,
  VarInt:$num_columns
)>;

// The getChecked without a context takes it from the return type, which is
// null when the function returns nothing.
def FuncType : CheckedDialectType<(type
  Array<Type>:$inputs,
  OptionalType:$optionalReturnType,
  Bool:$varArg
), "llvm::ArrayRef(inputs), optionalReturnType, varArg">;

def MethodType : DialectType<(type
  Ty<"cir::FuncType">:$member_func_ty,
  Ty<"cir::RecordType">:$class_ty
)>;

def VoidType : DialectType<(type)>;

def BitFieldType : CheckedDialectType<(type
  OptionalType:$storage_type,
  Array<List<Attr<"cir::BitFieldDeclAttr">>>:$fields
), "storage_type, llvm::ArrayRef(fields)">;

def StructType : RecordDialectType;
def UnionType : RecordDialectType;

def EhTokenType : DialectType<(type)>;
def CleanupTokenType : DialectType<(type)>;
def CatchTokenType : DialectType<(type)>;

def CIRDialectTypes : DialectTypes<"CIR"> {
  let elems = [
    IntType,
    SingleType,
    DoubleType,
    FP16Type,
    BF16Type,
    FP80Type,
    FP128Type,
    LongDoubleType,
    ComplexType,
    PointerType,
    DataMemberType,
    VPtrType,
    CUDADeviceSurfaceType,
    CUDADeviceTextureType,
    BoolType,
    ArrayType,
    VectorType,
    MatrixType,
    FuncType,
    MethodType,
    VoidType,
    BitFieldType,
    StructType,
    UnionType,
    EhTokenType,
    CleanupTokenType,
    CatchTokenType,
    // APPEND ONLY, as for the attributes above.
  ];
}

#endif // CIR_DIALECT_BYTECODE
