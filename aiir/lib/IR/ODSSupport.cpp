//===- ODSSupport.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains out-of-line implementations of the support types that
// Operation and related classes build on top of.
//
//===----------------------------------------------------------------------===//

#include "aiir/IR/ODSSupport.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Diagnostics.h"

using namespace aiir;

LogicalResult
aiir::convertFromAttribute(int64_t &storage, Attribute attr,
                           function_ref<InFlightDiagnostic()> emitError) {
  auto valueAttr = dyn_cast<IntegerAttr>(attr);
  if (!valueAttr) {
    emitError() << "expected IntegerAttr for key `value`";
    return failure();
  }
  storage = valueAttr.getValue().getSExtValue();
  return success();
}
Attribute aiir::convertToAttribute(AIIRContext *ctx, int64_t storage) {
  return IntegerAttr::get(IntegerType::get(ctx, 64), storage);
}

LogicalResult
aiir::convertFromAttribute(int32_t &storage, Attribute attr,
                           function_ref<InFlightDiagnostic()> emitError) {
  auto valueAttr = dyn_cast<IntegerAttr>(attr);
  if (!valueAttr) {
    emitError() << "expected IntegerAttr for key `value`";
    return failure();
  }
  storage = valueAttr.getValue().getSExtValue();
  return success();
}
Attribute aiir::convertToAttribute(AIIRContext *ctx, int32_t storage) {
  return IntegerAttr::get(IntegerType::get(ctx, 32), storage);
}

LogicalResult
aiir::convertFromAttribute(int8_t &storage, Attribute attr,
                           function_ref<InFlightDiagnostic()> emitError) {
  auto valueAttr = dyn_cast<IntegerAttr>(attr);
  if (!valueAttr) {
    emitError() << "expected IntegerAttr for key `value`";
    return failure();
  }
  storage = valueAttr.getValue().getSExtValue();
  return success();
}

Attribute aiir::convertToAttribute(AIIRContext *ctx, int8_t storage) {
  /// Convert the provided int8_t to an IntegerAttr attribute.
  return IntegerAttr::get(IntegerType::get(ctx, 8), storage);
}

LogicalResult
aiir::convertFromAttribute(uint8_t &storage, Attribute attr,
                           function_ref<InFlightDiagnostic()> emitError) {
  auto valueAttr = dyn_cast<IntegerAttr>(attr);
  if (!valueAttr) {
    emitError() << "expected IntegerAttr for key `value`";
    return failure();
  }
  storage = valueAttr.getValue().getZExtValue();
  return success();
}

Attribute aiir::convertToAttribute(AIIRContext *ctx, uint8_t storage) {
  /// Convert the provided uint8_t to an IntegerAttr attribute.
  return IntegerAttr::get(IntegerType::get(ctx, 8), storage);
}

LogicalResult
aiir::convertFromAttribute(std::string &storage, Attribute attr,
                           function_ref<InFlightDiagnostic()> emitError) {
  auto valueAttr = dyn_cast<StringAttr>(attr);
  if (!valueAttr)
    return emitError()
           << "expected string property to come from string attribute";
  storage = valueAttr.getValue().str();
  return success();
}
Attribute aiir::convertToAttribute(AIIRContext *ctx,
                                   const std::string &storage) {
  return StringAttr::get(ctx, storage);
}

LogicalResult
aiir::convertFromAttribute(bool &storage, Attribute attr,
                           function_ref<InFlightDiagnostic()> emitError) {
  auto valueAttr = dyn_cast<BoolAttr>(attr);
  if (!valueAttr)
    return emitError()
           << "expected string property to come from string attribute";
  storage = valueAttr.getValue();
  return success();
}
Attribute aiir::convertToAttribute(AIIRContext *ctx, bool storage) {
  return BoolAttr::get(ctx, storage);
}

template <typename DenseArrayTy, typename T>
static LogicalResult
convertDenseArrayFromAttr(MutableArrayRef<T> storage, Attribute attr,
                          function_ref<InFlightDiagnostic()> emitError,
                          StringRef denseArrayTyStr) {
  auto valueAttr = dyn_cast<DenseArrayTy>(attr);
  if (!valueAttr) {
    emitError() << "expected " << denseArrayTyStr << " for key `value`";
    return failure();
  }
  if (valueAttr.size() != static_cast<int64_t>(storage.size())) {
    emitError() << "size mismatch in attribute conversion: " << valueAttr.size()
                << " vs " << storage.size();
    return failure();
  }
  llvm::copy(valueAttr.asArrayRef(), storage.begin());
  return success();
}
LogicalResult
aiir::convertFromAttribute(MutableArrayRef<int64_t> storage, Attribute attr,
                           function_ref<InFlightDiagnostic()> emitError) {
  return convertDenseArrayFromAttr<DenseI64ArrayAttr>(storage, attr, emitError,
                                                      "DenseI64ArrayAttr");
}
LogicalResult
aiir::convertFromAttribute(MutableArrayRef<int32_t> storage, Attribute attr,
                           function_ref<InFlightDiagnostic()> emitError) {
  return convertDenseArrayFromAttr<DenseI32ArrayAttr>(storage, attr, emitError,
                                                      "DenseI32ArrayAttr");
}

template <typename DenseArrayTy, typename T>
static LogicalResult
convertDenseArrayFromAttr(SmallVectorImpl<T> &storage, Attribute attr,
                          function_ref<InFlightDiagnostic()> emitError,
                          StringRef denseArrayTyStr) {
  auto valueAttr = dyn_cast<DenseArrayTy>(attr);
  if (!valueAttr) {
    emitError() << "expected " << denseArrayTyStr << " for key `value`";
    return failure();
  }
  storage.resize_for_overwrite(valueAttr.size());
  llvm::copy(valueAttr.asArrayRef(), storage.begin());
  return success();
}
LogicalResult
aiir::convertFromAttribute(SmallVectorImpl<int64_t> &storage, Attribute attr,
                           function_ref<InFlightDiagnostic()> emitError) {
  return convertDenseArrayFromAttr<DenseI64ArrayAttr>(storage, attr, emitError,
                                                      "DenseI64ArrayAttr");
}
LogicalResult
aiir::convertFromAttribute(SmallVectorImpl<int32_t> &storage, Attribute attr,
                           function_ref<InFlightDiagnostic()> emitError) {
  return convertDenseArrayFromAttr<DenseI32ArrayAttr>(storage, attr, emitError,
                                                      "DenseI32ArrayAttr");
}

Attribute aiir::convertToAttribute(AIIRContext *ctx,
                                   ArrayRef<int64_t> storage) {
  return DenseI64ArrayAttr::get(ctx, storage);
}

Attribute aiir::convertToAttribute(AIIRContext *ctx,
                                   ArrayRef<int32_t> storage) {
  return DenseI32ArrayAttr::get(ctx, storage);
}
