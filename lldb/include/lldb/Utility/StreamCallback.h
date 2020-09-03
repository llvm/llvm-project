//===-- StreamCallback.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_STREAMCALLBACK_H
#define LLDB_UTILITY_STREAMCALLBACK_H

#include "lldb/lldb-types.h"
#include "llvm/Support/raw_ostream.h"

#include <stddef.h>
#include <stdint.h>

namespace lldb_private {

class StreamCallback : public llvm::raw_ostream {
public:
  StreamCallback(lldb::LogOutputCallback callback, void *baton);
  ~StreamCallback() override = default;

private:
  lldb::LogOutputCallback m_callback;
  void *m_baton;

  void write_impl(const char *Ptr, size_t Size) override;
  uint64_t current_pos() const override;
};

} // namespace lldb_private

#endif // LLDB_UTILITY_STREAMCALLBACK_H
