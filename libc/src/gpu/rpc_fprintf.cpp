//===-- GPU implementation of fprintf -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "rpc_fprintf.h"

#include "src/__support/CPP/string_view.h"
#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/__support/common.h"
#include "src/stdio/gpu/file.h"

namespace LIBC_NAMESPACE {

template <uint16_t opcode>
int fprintf_impl(::FILE *__restrict file, const char *__restrict format,
                 size_t format_size, void *args, size_t args_size) {
  uint64_t mask = gpu::get_lane_mask();
  rpc::Client::Port port = rpc::client.open<opcode>();

  if constexpr (opcode == RPC_PRINTF_TO_STREAM) {
    port.send([&](rpc::Buffer *buffer) {
      buffer->data[0] = reinterpret_cast<uintptr_t>(file);
    });
  }

  port.send_n(format, format_size);
  port.send_n(args, args_size);

  uint32_t ret = 0;
  for (;;) {
    const char *str = nullptr;
    port.recv([&](rpc::Buffer *buffer) {
      ret = static_cast<uint32_t>(buffer->data[0]);
      str = reinterpret_cast<const char *>(buffer->data[1]);
    });
    // If any lanes have a string argument it needs to be copied back.
    if (!gpu::ballot(mask, str))
      break;

    uint64_t size = str ? internal::string_length(str) + 1 : 0;
    port.send_n(str, size);
  }

  port.close();
  return ret;
}

// TODO: This is a stand-in function that uses a struct pointer and size in
// place of varargs. Once varargs support is added we will use that to
// implement the real version.
LLVM_LIBC_FUNCTION(int, rpc_fprintf,
                   (::FILE *__restrict stream, const char *__restrict format,
                    void *args, size_t size)) {
  cpp::string_view str(format);
  if (stream == stdout)
    return fprintf_impl<RPC_PRINTF_TO_STDOUT>(stream, format, str.size() + 1,
                                              args, size);
  else if (stream == stderr)
    return fprintf_impl<RPC_PRINTF_TO_STDERR>(stream, format, str.size() + 1,
                                              args, size);
  else
    return fprintf_impl<RPC_PRINTF_TO_STREAM>(stream, format, str.size() + 1,
                                              args, size);
}

} // namespace LIBC_NAMESPACE
