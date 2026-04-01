//===- Utils.h - C API General Utilities ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines general utilities for C API. This file should not be
// included from C++ code other than C API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CAPI_UTILS_H
#define AIIR_CAPI_UTILS_H

#include <utility>

#include "aiir-c/Support.h"
#include "llvm/Support/raw_ostream.h"

//===----------------------------------------------------------------------===//
// Printing helper.
//===----------------------------------------------------------------------===//

namespace aiir {
namespace detail {
/// A simple raw ostream subclass that forwards write_impl calls to the
/// user-supplied callback together with opaque user-supplied data.
class CallbackOstream : public llvm::raw_ostream {
public:
  CallbackOstream(std::function<void(AiirStringRef, void *)> callback,
                  void *opaqueData)
      : raw_ostream(/*unbuffered=*/true), callback(std::move(callback)),
        opaqueData(opaqueData), pos(0u) {}

  void write_impl(const char *ptr, size_t size) override {
    AiirStringRef string = aiirStringRefCreate(ptr, size);
    callback(string, opaqueData);
    pos += size;
  }

  uint64_t current_pos() const override { return pos; }

private:
  std::function<void(AiirStringRef, void *)> callback;
  void *opaqueData;
  uint64_t pos;
};
} // namespace detail
} // namespace aiir

#endif // AIIR_CAPI_UTILS_H
