//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRTYPES_H
#define CLANG_CIR_DIALECT_IR_CIRTYPES_H

#include "aiir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "aiir/IR/Attributes.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/IR/Types.h"
#include "aiir/Interfaces/DataLayoutInterfaces.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Interfaces/CIRTypeInterfaces.h"

namespace cir {

namespace detail {
struct RecordTypeStorage;
} // namespace detail

bool isValidFundamentalIntWidth(unsigned width);

/// Returns true if the type is a CIR sized type.
///
/// Types are sized if they implement SizedTypeInterface and
/// return true from its method isSized.
///
/// Unsized types are those that do not have a size, such as
/// void, or abstract types.
bool isSized(aiir::Type ty);

//===----------------------------------------------------------------------===//
// AddressSpace helpers
//===----------------------------------------------------------------------===//

cir::LangAddressSpace toCIRLangAddressSpace(clang::LangAS langAS);

// Compare a CIR memory space attribute with a Clang LangAS.
bool isMatchingAddressSpace(aiir::ptr::MemorySpaceAttrInterface cirAS,
                            clang::LangAS as);

/// Convert an AST LangAS to the appropriate CIR address space attribute
/// interface.
aiir::ptr::MemorySpaceAttrInterface
toCIRAddressSpaceAttr(aiir::AIIRContext &ctx, clang::LangAS langAS);

/// Normalize LangAddressSpace::Default to null (empty attribute).
aiir::ptr::MemorySpaceAttrInterface
normalizeDefaultAddressSpace(aiir::ptr::MemorySpaceAttrInterface addrSpace);

bool isSupportedCIRMemorySpaceAttr(
    aiir::ptr::MemorySpaceAttrInterface memorySpace);

} // namespace cir

//===----------------------------------------------------------------------===//
// CIR Dialect Tablegen'd Types
//===----------------------------------------------------------------------===//

namespace cir {

#include "clang/CIR/Dialect/IR/CIRTypeConstraints.h.inc"

} // namespace cir

#define GET_TYPEDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsTypes.h.inc"

#endif // CLANG_CIR_DIALECT_IR_CIRTYPES_H
