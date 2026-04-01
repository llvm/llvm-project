//===- LLVMDialect.h - AIIR LLVM IR dialect ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LLVM IR dialect in AIIR, containing LLVM operations and
// LLVM type system.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_LLVMIR_LLVMATTRS_H_
#define AIIR_DIALECT_LLVMIR_LLVMATTRS_H_

#include "aiir/Dialect/LLVMIR/LLVMTypes.h"
#include "aiir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/Interfaces/DataLayoutInterfaces.h"
#include <optional>

#include "aiir/Dialect/LLVMIR/LLVMOpsEnums.h.inc"

namespace aiir {
namespace LLVM {

/// This class represents the base attribute for all debug info attributes.
class DINodeAttr : public Attribute {
public:
  using Attribute::Attribute;

  // Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// This class represents a LLVM attribute that describes a debug info scope.
class DIScopeAttr : public DINodeAttr {
public:
  using DINodeAttr::DINodeAttr;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// This class represents a LLVM attribute that describes a local debug info
/// scope.
class DILocalScopeAttr : public DIScopeAttr {
public:
  using DIScopeAttr::DIScopeAttr;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// This class represents a LLVM attribute that describes a debug info type.
class DITypeAttr : public DINodeAttr {
public:
  using DINodeAttr::DINodeAttr;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// This class represents a LLVM attribute that describes a debug info variable.
class DIVariableAttr : public DINodeAttr {
public:
  using DINodeAttr::DINodeAttr;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// Base class for LLVM attributes participating in the TBAA graph.
class TBAANodeAttr : public Attribute {
public:
  using Attribute::Attribute;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);

  /// Required by DenseMapInfo to create empty and tombstone key.
  static TBAANodeAttr getFromOpaquePointer(const void *pointer) {
    return TBAANodeAttr(reinterpret_cast<const ImplType *>(pointer));
  }
};

// Inline the LLVM generated Linkage enum and utility.
// This is only necessary to isolate the "enum generated code" from the
// attribute definition itself.
// TODO: this shouldn't be needed after we unify the attribute generation, i.e.
// --gen-attr-* and --gen-attrdef-*.
using cconv::CConv;
using linkage::Linkage;
using tailcallkind::TailCallKind;

namespace detail {
/// Checks whether the given type is an LLVM type that can be loaded or stored.
bool isValidLoadStoreImpl(Type type, ptr::AtomicOrdering ordering,
                          std::optional<int64_t> alignment,
                          const ::aiir::DataLayout *dataLayout,
                          function_ref<InFlightDiagnostic()> emitError);
} // namespace detail
} // namespace LLVM
} // namespace aiir

#include "aiir/Dialect/LLVMIR/LLVMAttrInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/LLVMIR/LLVMOpsAttrDefs.h.inc"

#endif // AIIR_DIALECT_LLVMIR_LLVMATTRS_H_
