//===- TestOps.h - AIIR Test Dialect Operations ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TESTOPS_H
#define AIIR_TESTOPS_H

#include "TestAttributes.h"
#include "TestInterfaces.h"
#include "TestTypes.h"
#include "aiir/Bytecode/BytecodeImplementation.h"
#include "aiir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Dialect/DLTI/Traits.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/LLVMIR/NVVMRequiresSMTraits.h"
#include "aiir/Dialect/Linalg/IR/Linalg.h"
#include "aiir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "aiir/Dialect/Traits.h"
#include "aiir/IR/AsmState.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/DialectResourceBlobManager.h"
#include "aiir/IR/ExtensibleDialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/RegionKindInterface.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/CallInterfaces.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "aiir/Interfaces/DerivedAttributeOpInterface.h"
#include "aiir/Interfaces/InferIntRangeInterface.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/LoopLikeInterface.h"
#include "aiir/Interfaces/MemorySlotInterfaces.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "aiir/Interfaces/ValueBoundsOpInterface.h"
#include "aiir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace test {
class TestDialect;

//===----------------------------------------------------------------------===//
// TestResource
//===----------------------------------------------------------------------===//

/// A test resource for side effects (under DefaultResource).
struct TestResource : public aiir::SideEffects::Resource::Base<
                          TestResource, aiir::SideEffects::DefaultResource> {
  llvm::StringRef getName() const final { return "<Test>"; }
  aiir::SideEffects::Resource *getParent() const override {
    return aiir::SideEffects::DefaultResource::get();
  }
};

/// A test resource that is a root (disjoint from DefaultResource).
struct TestNonAddressableResource
    : public aiir::SideEffects::Resource::Base<TestNonAddressableResource> {
  llvm::StringRef getName() const final { return "<TestNonAddressable>"; }
  bool isAddressable() const override { return false; }
};

/// Two disjoint sub-resources (roots) for testing sibling disjointness.
struct TestNonAddressableSubResourceA
    : public aiir::SideEffects::Resource::Base<TestNonAddressableSubResourceA> {
  TestNonAddressableSubResourceA() = default;
  llvm::StringRef getName() const override {
    return "TestNonAddressableSubResourceA";
  }
  bool isAddressable() const override { return false; }

protected:
  TestNonAddressableSubResourceA(aiir::TypeID id) : Base(id) {}
};

struct TestNonAddressableSubResourceB
    : public aiir::SideEffects::Resource::Base<TestNonAddressableSubResourceB> {
  TestNonAddressableSubResourceB() = default;
  llvm::StringRef getName() const override {
    return "TestNonAddressableSubResourceB";
  }
  bool isAddressable() const override { return false; }

protected:
  TestNonAddressableSubResourceB(aiir::TypeID id) : Base(id) {}
};

struct TestNonAddressableResourceA
    : public aiir::SideEffects::Resource::Base<TestNonAddressableResourceA,
                                               TestNonAddressableSubResourceA> {
  llvm::StringRef getName() const final { return "<TestNonAddressableA>"; }
  bool isAddressable() const override { return false; }
  aiir::SideEffects::Resource *getParent() const override {
    return TestNonAddressableSubResourceA::get();
  }
};

struct TestNonAddressableResourceB
    : public aiir::SideEffects::Resource::Base<TestNonAddressableResourceB,
                                               TestNonAddressableSubResourceB> {
  llvm::StringRef getName() const final { return "<TestNonAddressableB>"; }
  bool isAddressable() const override { return false; }
  aiir::SideEffects::Resource *getParent() const override {
    return TestNonAddressableSubResourceB::get();
  }
};

//===----------------------------------------------------------------------===//
// PropertiesWithCustomPrint
//===----------------------------------------------------------------------===//

struct PropertiesWithCustomPrint {
  /// A shared_ptr to a const object is safe: it is equivalent to a value-based
  /// member. Here the label will be deallocated when the last operation
  /// refering to it is destroyed. However there is no pool-allocation: this is
  /// offloaded to the client.
  std::shared_ptr<const std::string> label;
  int value;
  bool operator==(const PropertiesWithCustomPrint &rhs) const {
    return value == rhs.value && *label == *rhs.label;
  }
};

llvm::LogicalResult setPropertiesFromAttribute(
    PropertiesWithCustomPrint &prop, aiir::Attribute attr,
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError);
aiir::DictionaryAttr
getPropertiesAsAttribute(aiir::AIIRContext *ctx,
                         const PropertiesWithCustomPrint &prop);
llvm::hash_code computeHash(const PropertiesWithCustomPrint &prop);
void customPrintProperties(aiir::OpAsmPrinter &p,
                           const PropertiesWithCustomPrint &prop);
aiir::ParseResult customParseProperties(aiir::OpAsmParser &parser,
                                        PropertiesWithCustomPrint &prop);

//===----------------------------------------------------------------------===//
// MyPropStruct
//===----------------------------------------------------------------------===//
namespace test_properties {
class MyPropStruct {
public:
  std::string content;
  // These three methods are invoked through the  `MyStructProperty` wrapper
  // defined in TestOps.td
  aiir::Attribute asAttribute(aiir::AIIRContext *ctx) const;
  static llvm::LogicalResult
  setFromAttr(MyPropStruct &prop, aiir::Attribute attr,
              llvm::function_ref<aiir::InFlightDiagnostic()> emitError);
  llvm::hash_code hash() const;
  bool operator==(const MyPropStruct &rhs) const {
    return content == rhs.content;
  }
};
inline llvm::hash_code hash_value(const MyPropStruct &S) { return S.hash(); }
} // namespace test_properties
using test_properties::MyPropStruct;

llvm::LogicalResult readFromAiirBytecode(aiir::DialectBytecodeReader &reader,
                                         MyPropStruct &prop);
void writeToAiirBytecode(aiir::DialectBytecodeWriter &writer,
                         MyPropStruct &prop);

//===----------------------------------------------------------------------===//
// VersionedProperties
//===----------------------------------------------------------------------===//

struct VersionedProperties {
  // For the sake of testing, assume that this object was associated to version
  // 1.2 of the test dialect when having only one int value. In the current
  // version 2.0, the property has two values. We also assume that the class is
  // upgrade-able if value2 = 0.
  int value1;
  int value2;
  bool operator==(const VersionedProperties &rhs) const {
    return value1 == rhs.value1 && value2 == rhs.value2;
  }
};

llvm::LogicalResult setPropertiesFromAttribute(
    VersionedProperties &prop, aiir::Attribute attr,
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError);
aiir::DictionaryAttr getPropertiesAsAttribute(aiir::AIIRContext *ctx,
                                              const VersionedProperties &prop);
llvm::hash_code computeHash(const VersionedProperties &prop);
void customPrintProperties(aiir::OpAsmPrinter &p,
                           const VersionedProperties &prop);
aiir::ParseResult customParseProperties(aiir::OpAsmParser &parser,
                                        VersionedProperties &prop);

//===----------------------------------------------------------------------===//
// Bytecode Support
//===----------------------------------------------------------------------===//

llvm::LogicalResult readFromAiirBytecode(aiir::DialectBytecodeReader &reader,
                                         llvm::MutableArrayRef<int64_t> prop);
void writeToAiirBytecode(aiir::DialectBytecodeWriter &writer,
                         llvm::ArrayRef<int64_t> prop);

} // namespace test

#define GET_OP_CLASSES
#include "TestOps.h.inc"

#endif // AIIR_TESTOPS_H
