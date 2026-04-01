//===- TestTypes.h - AIIR Test Dialect Types --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains types defined by the TestDialect for testing various
// features of AIIR.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TESTTYPES_H
#define AIIR_TESTTYPES_H

#include <optional>
#include <tuple>

#include "TestTraits.h"
#include "aiir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.h"
#include "aiir/IR/BuiltinTypeInterfaces.h"
#include "aiir/IR/Diagnostics.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/DialectImplementation.h"
#include "aiir/IR/Operation.h"
#include "aiir/IR/Types.h"
#include "aiir/Interfaces/DataLayoutInterfaces.h"

namespace test {
class TestAttrWithFormatAttr;

/// FieldInfo represents a field in the StructType data type. It is used as a
/// parameter in TestTypeDefs.td.
struct FieldInfo {
  llvm::StringRef name;
  aiir::Type type;

  // Custom allocation called from generated constructor code
  FieldInfo allocateInto(aiir::TypeStorageAllocator &alloc) const {
    return FieldInfo{alloc.copyInto(name), type};
  }
};

/// A custom type for a test type parameter.
struct CustomParam {
  int value;

  bool operator==(const CustomParam &other) const {
    return other.value == value;
  }
};

inline llvm::hash_code hash_value(const test::CustomParam &param) {
  return llvm::hash_value(param.value);
}

} // namespace test

namespace aiir {
template <>
struct FieldParser<test::CustomParam> {
  static FailureOr<test::CustomParam> parse(AsmParser &parser) {
    auto value = FieldParser<int>::parse(parser);
    if (failed(value))
      return failure();
    return test::CustomParam{*value};
  }
};

inline aiir::AsmPrinter &operator<<(aiir::AsmPrinter &printer,
                                    test::CustomParam param) {
  return printer << param.value;
}

/// Overload the attribute parameter parser for optional integers.
template <>
struct FieldParser<std::optional<int>> {
  static FailureOr<std::optional<int>> parse(AsmParser &parser) {
    std::optional<int> value;
    value.emplace();
    OptionalParseResult result = parser.parseOptionalInteger(*value);
    if (result.has_value()) {
      if (succeeded(*result))
        return value;
      return failure();
    }
    value.reset();
    return value;
  }
};
} // namespace aiir

#include "TestTypeInterfaces.h.inc"

namespace test {

/// Storage for simple named recursive types, where the type is identified by
/// its name and can "contain" another type, including itself.
struct TestRecursiveTypeStorage : public ::aiir::TypeStorage {
  using KeyTy = ::llvm::StringRef;

  explicit TestRecursiveTypeStorage(::llvm::StringRef key) : name(key) {}

  bool operator==(const KeyTy &other) const { return name == other; }

  static TestRecursiveTypeStorage *
  construct(::aiir::TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<TestRecursiveTypeStorage>())
        TestRecursiveTypeStorage(allocator.copyInto(key));
  }

  ::llvm::LogicalResult mutate(::aiir::TypeStorageAllocator &allocator,
                               ::aiir::Type newBody) {
    // Cannot set a different body than before.
    if (body && body != newBody)
      return ::aiir::failure();

    body = newBody;
    return ::aiir::success();
  }

  ::llvm::StringRef name;
  ::aiir::Type body;
};

/// Simple recursive type identified by its name and pointing to another named
/// type, potentially itself. This requires the body to be mutated separately
/// from type creation.
class TestRecursiveType
    : public ::aiir::Type::TypeBase<TestRecursiveType, ::aiir::Type,
                                    TestRecursiveTypeStorage,
                                    ::aiir::TypeTrait::IsMutable> {
public:
  using Base::Base;

  static constexpr ::aiir::StringLiteral name = "test.recursive";

  static TestRecursiveType get(::aiir::AIIRContext *ctx,
                               ::llvm::StringRef name) {
    return Base::get(ctx, name);
  }

  /// Body getter and setter.
  ::llvm::LogicalResult setBody(Type body) { return Base::mutate(body); }
  ::aiir::Type getBody() const { return getImpl()->body; }

  /// Name/key getter.
  ::llvm::StringRef getName() { return getImpl()->name; }
};

} // namespace test

#define GET_TYPEDEF_CLASSES
#include "TestTypeDefs.h.inc"

#endif // AIIR_TESTTYPES_H
