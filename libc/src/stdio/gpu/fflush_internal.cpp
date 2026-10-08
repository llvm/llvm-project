//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// GPU implementation of the internal flushing helpers.
///
//===----------------------------------------------------------------------===//

#include "src/stdio/fflush_internal.h"

#include "file.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/FILE.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

static int flush_on_host(::FILE *stream) {
  int ret;
  rpc::Client::Port port = rpc::client.open<LIBC_FFLUSH>();
  port.send_and_recv(
      [=](rpc::Buffer *buffer, uint32_t) {
        buffer->data[0] = file::from_stream(stream);
      },
      [&](rpc::Buffer *buffer, uint32_t) {
        ret = static_cast<int>(buffer->data[0]);
      });
  return ret;
}

ErrorOr<int> flush_stream(::FILE *stream) { return flush_on_host(stream); }

ErrorOr<int> flush_all_streams() { return flush_on_host(nullptr); }

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
