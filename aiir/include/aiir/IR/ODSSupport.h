//===- ODSSupport.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a number of support method for ODS generated code.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_IR_ODSSUPPORT_H
#define AIIR_IR_ODSSUPPORT_H

#include "aiir/IR/Attributes.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/Support/LLVM.h"

namespace aiir {

//===----------------------------------------------------------------------===//
// Support for properties
//===----------------------------------------------------------------------===//

/// Convert an IntegerAttr attribute to an int64_t, or return an error if the
/// attribute isn't an IntegerAttr. If the optional diagnostic is provided an
/// error message is also emitted.
LogicalResult
convertFromAttribute(int64_t &storage, Attribute attr,
                     function_ref<InFlightDiagnostic()> emitError);

/// Convert the provided int64_t to an IntegerAttr attribute.
Attribute convertToAttribute(AIIRContext *ctx, int64_t storage);

/// Convert an IntegerAttr attribute to an int32_t, or return an error if the
/// attribute isn't an IntegerAttr. If the optional diagnostic is provided an
/// error message is also emitted.
LogicalResult
convertFromAttribute(int32_t &storage, Attribute attr,
                     function_ref<InFlightDiagnostic()> emitError);

/// Convert the provided int32_t to an IntegerAttr attribute.
Attribute convertToAttribute(AIIRContext *ctx, int32_t storage);

/// Convert an IntegerAttr attribute to an int8_t, or return an error if the
/// attribute isn't an IntegerAttr. If the optional diagnostic is provided an
/// error message is also emitted.
LogicalResult
convertFromAttribute(int8_t &storage, Attribute attr,
                     function_ref<InFlightDiagnostic()> emitError);

/// Convert the provided int8_t to an IntegerAttr attribute.
Attribute convertToAttribute(AIIRContext *ctx, int8_t storage);

/// Convert an IntegerAttr attribute to an uint8_t, or return an error if the
/// attribute isn't an IntegerAttr. If the optional diagnostic is provided an
/// error message is also emitted.
LogicalResult
convertFromAttribute(uint8_t &storage, Attribute attr,
                     function_ref<InFlightDiagnostic()> emitError);

/// Convert the provided uint8_t to an IntegerAttr attribute.
Attribute convertToAttribute(AIIRContext *ctx, uint8_t storage);

/// Extract the string from `attr` into `storage`. If `attr` is not a
/// `StringAttr`, return failure and emit an error into the diagnostic from
/// `emitError`.
LogicalResult
convertFromAttribute(std::string &storage, Attribute attr,
                     function_ref<InFlightDiagnostic()> emitError);

/// Convert the given string into a StringAttr. Note that this takes a reference
/// to the storage of a string property, which is an std::string.
Attribute convertToAttribute(AIIRContext *ctx, const std::string &storage);

/// Extract the boolean from `attr` into `storage`. If `attr` is not a
/// `BoolAttr`, return failure and emit an error into the diagnostic from
/// `emitError`.
LogicalResult
convertFromAttribute(bool &storage, Attribute attr,
                     function_ref<InFlightDiagnostic()> emitError);

/// Convert the given string into a BooleanAttr.
Attribute convertToAttribute(AIIRContext *ctx, bool storage);

/// Convert a DenseI64ArrayAttr to the provided storage. It is expected that the
/// storage has the same size as the array. An error is returned if the
/// attribute isn't a DenseI64ArrayAttr or it does not have the same size. If
/// the optional diagnostic is provided an error message is also emitted.
LogicalResult
convertFromAttribute(MutableArrayRef<int64_t> storage, Attribute attr,
                     function_ref<InFlightDiagnostic()> emitError);

/// Convert a DenseI32ArrayAttr to the provided storage. It is expected that the
/// storage has the same size as the array. An error is returned if the
/// attribute isn't a DenseI32ArrayAttr or it does not have the same size. If
/// the optional diagnostic is provided an error message is also emitted.
LogicalResult
convertFromAttribute(MutableArrayRef<int32_t> storage, Attribute attr,
                     function_ref<InFlightDiagnostic()> emitError);

/// Convert a DenseI64ArrayAttr to the provided storage, which will be
/// cleared before writing. An error is returned and emitted to the optional
/// `emitError` function if the attribute isn't a DenseI64ArrayAttr.
LogicalResult
convertFromAttribute(SmallVectorImpl<int64_t> &storage, Attribute attr,
                     function_ref<InFlightDiagnostic()> emitError);

/// Convert a DenseI32ArrayAttr to the provided storage, which will be
/// cleared before writing. It is expected that the storage has the same size as
/// the array. An error is returned and emitted to the optional `emitError`
/// function if the attribute isn't a DenseI32ArrayAttr.
LogicalResult
convertFromAttribute(SmallVectorImpl<int32_t> &storage, Attribute attr,
                     function_ref<InFlightDiagnostic()> emitError);

/// Convert the provided ArrayRef<int64_t> to a DenseI64ArrayAttr attribute.
Attribute convertToAttribute(AIIRContext *ctx, ArrayRef<int64_t> storage);

/// Convert the provided ArrayRef<int32_t> to a DenseI32ArrayAttr attribute.
Attribute convertToAttribute(AIIRContext *ctx, ArrayRef<int32_t> storage);

} // namespace aiir

#endif // AIIR_IR_ODSSUPPORT_H
