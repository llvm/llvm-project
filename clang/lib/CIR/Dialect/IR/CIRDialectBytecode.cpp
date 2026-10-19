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

// Both namespaces declare a BoolAttr, named unqualified below. This must live
// in the inner namespace: at global scope it merges with the using-directives
// and stays ambiguous.
using cir::BoolAttr;

//===--------------------------------------------------------------------===//
// Helpers referenced by the generated readers
//===--------------------------------------------------------------------===//

// The payload width is implied by the type (mirroring BuiltinDialectBytecode)
// and recovered here. A mismatched type fails rather than asserts: it came
// from the same untrusted file, so it is a corrupt-input case.

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

// std::optional<T> parameters carry presence as a varint of its own ahead of
// the value, so the full width survives.
template <typename EntryTy>
static LogicalResult readOptionalInt(DialectBytecodeReader &reader,
                                     std::optional<EntryTy> &storage) {
  static_assert(std::is_unsigned_v<EntryTy>,
                "EntryTy must be unsigned: only unsigned varints are "
                "supported here, so a negative value cannot be represented");
  uint64_t present = 0;
  if (failed(reader.readVarInt(present)))
    return failure();
  if (present == 0) {
    storage = std::nullopt;
    return success();
  }
  if (present != 1)
    return reader.emitError() << "optional integer presence flag " << present
                              << " is neither 0 nor 1";
  uint64_t value = 0;
  if (failed(reader.readVarInt(value)))
    return failure();
  // Out-of-range values fail rather than truncating into a plausible index.
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
                "EntryTy must be unsigned: only unsigned varints are "
                "supported here, so a negative value cannot be represented");
  writer.writeVarInt(storage.has_value() ? 1 : 0);
  if (storage)
    writer.writeVarInt(*storage);
}

// std::optional<Attr> form (cir.method's symbol). A null attribute is not a
// valid state and normalizes to std::nullopt on the wire.

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

// A scoped enum validated through its symbolizer; out-of-range values fail
// instead of casting undefined values into the enum.
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

// A varint narrowed to a fixed-width integer; out-of-range values fail
// instead of truncating.
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

  Type readType(DialectBytecodeReader &reader) const override {
    return ::readType(getContext(), reader);
  }

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override {
    return ::writeType(type, writer);
  }
};
} // namespace

void cir::detail::addBytecodeInterface(CIRDialect *dialect) {
  dialect->addInterfaces<CIRDialectBytecodeInterface>();
}
