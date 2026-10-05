//===- CIRDialectBytecode.cpp - CIR Bytecode Implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRDialectBytecode.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Diagnostics.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include <limits>
#include <type_traits>

using namespace mlir;
using namespace cir;

namespace {

// Both namespaces opened above declare a BoolAttr, and the generated readers
// and writers name attributes unqualified. A using-declaration at global scope
// would not help: it merges with the names the using-directives inject there
// and stays ambiguous, so it has to live in this inner namespace, where it
// hides them.
using cir::BoolAttr;

//===--------------------------------------------------------------------===//
// Helpers referenced by the generated readers
//===--------------------------------------------------------------------===//

// IntAttr and FPAttr do not store their payload's width: it is already implied
// by the attribute's type, so storing it again would be redundant and would
// give a malformed file two disagreeing sources of truth. These recover the
// width from the type, mirroring the same helpers in BuiltinDialectBytecode.
//
// Both fail rather than assert on a type that is not the expected interface: a
// bytecode file is untrusted input, and the type here was read from that same
// file, so a mismatch is a corrupt-input case and not a bug.

static LogicalResult readAPIntWithKnownWidth(DialectBytecodeReader &reader,
                                             Type type, FailureOr<APInt> &val) {
  auto intType = mlir::dyn_cast<cir::IntTypeInterface>(type);
  if (!intType)
    return reader.emitError()
           << "expected a CIR integer type for cir.int, but got: " << type;
  val = reader.readAPIntWithKnownWidth(intType.getWidth());
  return success(succeeded(val));
}

static LogicalResult
readAPFloatWithKnownSemantics(DialectBytecodeReader &reader, Type type,
                              FailureOr<APFloat> &val) {
  auto fpType = mlir::dyn_cast<cir::FPTypeInterface>(type);
  if (!fpType)
    return reader.emitError()
           << "expected a CIR floating-point type for cir.fp, but got: "
           << type;
  val = reader.readAPFloatWithKnownSemantics(fpType.getFloatSemantics());
  return success(succeeded(val));
}

// An optional parameter spelled std::optional<T> rather than as a null-valued
// attribute needs its own presence flag. A null Attribute already means
// "absent", which is why OptionalAttribute in the .td needs none of this, but
// there is no null unsigned.
//
// writeVarIntWithFlag packs the flag into the varint's low bit, so an absent
// value costs one byte. These mirror the LLVM dialect's helpers of the same
// names (LLVMDialectBytecode.cpp).

// Unsigned only, and deliberately narrower than the LLVM dialect's version.
// writeVarIntWithFlag packs the presence bit into the low bit, i.e. it shifts
// the value left by one, so the channel is 63 bits and a negative EntryTy,
// which becomes a uint64_t with the top bit set, cannot round-trip at all.
// Rejecting signed types here beats discovering that at the first caller that
// has one.
template <typename EntryTy>
static LogicalResult readOptionalInt(DialectBytecodeReader &reader,
                                     std::optional<EntryTy> &storage) {
  static_assert(std::is_unsigned_v<EntryTy>,
                "EntryTy must be unsigned: writeVarIntWithFlag spends the low "
                "bit on the presence flag, so a negative value cannot be "
                "represented");
  uint64_t value = 0;
  bool present = false;
  if (failed(reader.readVarIntWithFlag(value, present)))
    return failure();
  if (!present) {
    storage = std::nullopt;
    return success();
  }
  // A file written by another version, or a corrupt one, can carry a value
  // this parameter cannot hold. Truncating it would turn that into a plausible
  // member index or byte offset; failing the read is what the width-sensitive
  // readers above (readAPIntWithKnownWidth, readAPFloatWithKnownSemantics) do,
  // and it is what the caller can actually act on.
  if (value > static_cast<uint64_t>(std::numeric_limits<EntryTy>::max()))
    return reader.emitError() << "optional integer value " << value
                              << " does not fit in destination type";
  storage = static_cast<EntryTy>(value);
  return success();
}

template <typename EntryTy>
static void writeOptionalInt(DialectBytecodeWriter &writer,
                             std::optional<EntryTy> storage) {
  static_assert(std::is_unsigned_v<EntryTy>,
                "EntryTy must be unsigned: writeVarIntWithFlag spends the low "
                "bit on the presence flag, so a negative value cannot be "
                "represented");
  // The channel is only 63 bits: values at or above 2^63 would lose the top
  // bit on the wire. No caller has one, and this is where that stops being
  // true loudly instead of silently.
  assert(!storage || *storage < (uint64_t(1) << 63));
  writer.writeVarIntWithFlag(storage.value_or(0), storage.has_value());
}

// The same, for a parameter that is an attribute but is still spelled
// std::optional<> (cir.method's symbol). An engaged optional holding a null
// attribute is not a valid state: it normalizes to std::nullopt on the wire,
// and the read side only ever produces an engaged optional holding a
// non-null attribute, or std::nullopt.

template <typename AttrTy>
static LogicalResult readStdOptionalAttribute(DialectBytecodeReader &reader,
                                              std::optional<AttrTy> &storage) {
  AttrTy attr;
  if (failed(reader.readOptionalAttribute(attr)))
    return failure();
  if (attr)
    storage = attr;
  else
    storage = std::nullopt;
  return success();
}

template <typename AttrTy>
static void writeStdOptionalAttribute(DialectBytecodeWriter &writer,
                                      std::optional<AttrTy> storage) {
  assert(!storage || *storage);
  writer.writeOptionalAttribute(storage.value_or(AttrTy()));
}

// A scoped enum read from an untrusted varint. The symbolizer rejects
// out-of-range values; without this a corrupt file would cast one straight
// into the enum, which is undefined behavior for a switch on the kind and an
// empty string from the printer's stringify.
template <typename EnumTy>
static LogicalResult readEnum(DialectBytecodeReader &reader, EnumTy &result,
                              std::optional<EnumTy> (*symbolize)(uint32_t),
                              llvm::StringRef enumName) {
  uint64_t value = 0;
  if (failed(reader.readVarInt(value)))
    return failure();
  if (value > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
    return reader.emitError()
           << enumName << " value " << value << " out of range";
  std::optional<EnumTy> kind = symbolize(static_cast<uint32_t>(value));
  if (!kind)
    return reader.emitError() << "invalid " << enumName << " value " << value;
  result = *kind;
  return success();
}

// A varint on the wire narrowed to a fixed-width integer. Mirror of
// readOptionalInt's bound check: a value the parameter cannot hold fails the
// read instead of truncating into a plausible count or index.
template <typename IntTy>
static LogicalResult readCheckedInt(DialectBytecodeReader &reader,
                                    IntTy &result) {
  static_assert(std::is_integral_v<IntTy> && !std::is_same_v<IntTy, bool>);
  if constexpr (std::is_signed_v<IntTy>) {
    int64_t value = 0;
    if (failed(reader.readSignedVarInt(value)))
      return failure();
    if (value > static_cast<int64_t>(std::numeric_limits<IntTy>::max()) ||
        value < static_cast<int64_t>(std::numeric_limits<IntTy>::min()))
      return reader.emitError() << "integer value " << value
                                << " does not fit in destination type";
    result = static_cast<IntTy>(value);
  } else {
    uint64_t value = 0;
    if (failed(reader.readVarInt(value)))
      return failure();
    if (value > static_cast<uint64_t>(std::numeric_limits<IntTy>::max()))
      return reader.emitError() << "integer value " << value
                                << " does not fit in destination type";
    result = static_cast<IntTy>(value);
  }
  return success();
}

//===--------------------------------------------------------------------===//
// Tablegen generated bytecode functions
//===--------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRDialectBytecode.cpp.inc"

//===--------------------------------------------------------------------===//
// CIRDialectBytecodeInterface
//===--------------------------------------------------------------------===//

/// Bytecode interface for the CIR dialect.
struct CIRDialectBytecodeInterface : public BytecodeDialectInterface {
  CIRDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  Attribute readAttribute(DialectBytecodeReader &reader) const override {
    return ::readAttribute(getContext(), reader);
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override {
    return ::writeAttribute(attr, writer);
  }
};
} // namespace

void cir::detail::addBytecodeInterface(CIRDialect *dialect) {
  dialect->addInterfaces<CIRDialectBytecodeInterface>();
}
