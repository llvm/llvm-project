//===-- Shared memory RPC server instantiation ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Workaround for missing __has_builtin in < GCC 10.
#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

// Make sure these are included first so they don't conflict with the system.
#include <limits.h>

#include "shared/rpc.h"
#include "shared/rpc_opcodes.h"

#include "src/__support/arg_list.h"
#include "src/stdio/printf_core/converter.h"
#include "src/stdio/printf_core/parser.h"
#include "src/stdio/printf_core/writer.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <variant>
#include <vector>

using namespace LIBC_NAMESPACE;
using namespace LIBC_NAMESPACE::printf_core;

namespace {
struct TempStorage {
  char *alloc(size_t size) {
    storage.emplace_back(std::make_unique<char[]>(size));
    return storage.back().get();
  }

  std::vector<std::unique_ptr<char[]>> storage;
};
} // namespace

enum Stream {
  File = 0,
  Stdin = 1,
  Stdout = 2,
  Stderr = 3,
};

// Get the associated stream out of an encoded number.
LIBC_INLINE ::FILE *to_stream(uintptr_t f) {
  ::FILE *stream = reinterpret_cast<FILE *>(f & ~0x3ull);
  Stream type = static_cast<Stream>(f & 0x3ull);
  if (type == Stdin)
    return stdin;
  if (type == Stdout)
    return stdout;
  if (type == Stderr)
    return stderr;
  return stream;
}

template <bool packed, uint32_t num_lanes>
static void handle_printf(rpc::Server::Port &port, TempStorage &temp_storage) {
  FILE *files[num_lanes] = {nullptr};
  // Get the appropriate output stream to use.
  if (port.get_opcode() == LIBC_PRINTF_TO_STREAM ||
      port.get_opcode() == LIBC_PRINTF_TO_STREAM_PACKED)
    port.recv([&](rpc::Buffer *buffer, uint32_t id) {
      files[id] = reinterpret_cast<FILE *>(buffer->data[0]);
    });
  else if (port.get_opcode() == LIBC_PRINTF_TO_STDOUT ||
           port.get_opcode() == LIBC_PRINTF_TO_STDOUT_PACKED)
    std::fill(files, files + num_lanes, stdout);
  else
    std::fill(files, files + num_lanes, stderr);

  uint64_t format_sizes[num_lanes] = {0};
  void *format[num_lanes] = {nullptr};

  uint64_t args_sizes[num_lanes] = {0};
  void *args[num_lanes] = {nullptr};

  // Recieve the format string and arguments from the client.
  port.recv_n(format, format_sizes,
              [&](uint64_t size) { return temp_storage.alloc(size); });

  // Parse the format string to get the expected size of the buffer.
  for (uint32_t lane = 0; lane < num_lanes; ++lane) {
    if (!format[lane])
      continue;

    WriteBuffer wb(nullptr, 0);
    Writer writer(&wb);

    internal::DummyArgList<packed> printf_args;
    Parser<internal::DummyArgList<packed> &> parser(
        reinterpret_cast<const char *>(format[lane]), printf_args);

    for (FormatSection cur_section = parser.get_next_section();
         !cur_section.raw_string.empty();
         cur_section = parser.get_next_section())
      ;
    args_sizes[lane] = printf_args.read_count();
  }
  port.send([&](rpc::Buffer *buffer, uint32_t id) {
    buffer->data[0] = args_sizes[id];
  });
  port.recv_n(args, args_sizes,
              [&](uint64_t size) { return temp_storage.alloc(size); });

  // Identify any arguments that are actually pointers to strings on the client.
  // Additionally we want to determine how much buffer space we need to print.
  std::vector<void *> strs_to_copy[num_lanes];
  int buffer_size[num_lanes] = {0};
  for (uint32_t lane = 0; lane < num_lanes; ++lane) {
    if (!format[lane])
      continue;

    WriteBuffer wb(nullptr, 0);
    Writer writer(&wb);

    internal::StructArgList<packed> printf_args(args[lane], args_sizes[lane]);
    Parser<internal::StructArgList<packed>> parser(
        reinterpret_cast<const char *>(format[lane]), printf_args);

    for (FormatSection cur_section = parser.get_next_section();
         !cur_section.raw_string.empty();
         cur_section = parser.get_next_section()) {
      if (cur_section.has_conv && cur_section.conv_name == 's' &&
          cur_section.conv_val_ptr) {
        strs_to_copy[lane].emplace_back(cur_section.conv_val_ptr);
        // Get the minimum size of the string in the case of padding.
        char c = '\0';
        cur_section.conv_val_ptr = &c;
        convert(&writer, cur_section);
      } else if (cur_section.has_conv) {
        // Ignore conversion errors for the first pass.
        convert(&writer, cur_section);
      } else {
        writer.write(cur_section.raw_string);
      }
    }
    buffer_size[lane] = writer.get_chars_written();
  }

  // Recieve any strings from the client and push them into a buffer.
  std::vector<void *> copied_strs[num_lanes];
  while (std::any_of(std::begin(strs_to_copy), std::end(strs_to_copy),
                     [](const auto &v) { return !v.empty() && v.back(); })) {
    port.send([&](rpc::Buffer *buffer, uint32_t id) {
      void *ptr = !strs_to_copy[id].empty() ? strs_to_copy[id].back() : nullptr;
      buffer->data[1] = reinterpret_cast<uintptr_t>(ptr);
      if (!strs_to_copy[id].empty())
        strs_to_copy[id].pop_back();
    });
    uint64_t str_sizes[num_lanes] = {0};
    void *strs[num_lanes] = {nullptr};
    port.recv_n(strs, str_sizes,
                [&](uint64_t size) { return temp_storage.alloc(size); });
    for (uint32_t lane = 0; lane < num_lanes; ++lane) {
      if (!strs[lane])
        continue;

      copied_strs[lane].emplace_back(strs[lane]);
      buffer_size[lane] += str_sizes[lane];
    }
  }

  // Perform the final formatting and printing using the LLVM C library printf.
  int results[num_lanes] = {0};
  for (uint32_t lane = 0; lane < num_lanes; ++lane) {
    if (!format[lane])
      continue;

    char *buffer = temp_storage.alloc(buffer_size[lane]);
    WriteBuffer wb(buffer, buffer_size[lane]);
    Writer writer(&wb);

    internal::StructArgList<packed> printf_args(args[lane], args_sizes[lane]);
    Parser<internal::StructArgList<packed>> parser(
        reinterpret_cast<const char *>(format[lane]), printf_args);

    // Parse and print the format string using the arguments we copied from
    // the client.
    int ret = 0;
    for (FormatSection cur_section = parser.get_next_section();
         !cur_section.raw_string.empty();
         cur_section = parser.get_next_section()) {
      // If this argument was a string we use the memory buffer we copied from
      // the client by replacing the raw pointer with the copied one.
      if (cur_section.has_conv && cur_section.conv_name == 's') {
        if (!copied_strs[lane].empty()) {
          cur_section.conv_val_ptr = copied_strs[lane].back();
          copied_strs[lane].pop_back();
        } else {
          cur_section.conv_val_ptr = nullptr;
        }
      }
      if (cur_section.has_conv) {
        ret = convert(&writer, cur_section);
        if (ret == -1)
          break;
      } else {
        writer.write(cur_section.raw_string);
      }
    }

    results[lane] = fwrite(buffer, 1, writer.get_chars_written(), files[lane]);
    if (results[lane] != writer.get_chars_written() || ret == -1)
      results[lane] = -1;
  }

  // Send the final return value and signal completion by setting the string
  // argument to null.
  port.send([&](rpc::Buffer *buffer, uint32_t id) {
    buffer->data[0] = static_cast<uint64_t>(results[id]);
    buffer->data[1] = reinterpret_cast<uintptr_t>(nullptr);
  });
}

template <uint32_t num_lanes>
rpc::Status handle_port_impl(rpc::Server::Port &port) {
  TempStorage temp_storage;

  switch (port.get_opcode()) {
  case LIBC_WRITE_TO_STREAM:
  case LIBC_WRITE_TO_STDERR:
  case LIBC_WRITE_TO_STDOUT:
  case LIBC_WRITE_TO_STDOUT_NEWLINE: {
    uint64_t sizes[num_lanes] = {0};
    void *strs[num_lanes] = {nullptr};
    FILE *files[num_lanes] = {nullptr};
    if (port.get_opcode() == LIBC_WRITE_TO_STREAM) {
      port.recv([&](rpc::Buffer *buffer, uint32_t id) {
        files[id] = reinterpret_cast<FILE *>(buffer->data[0]);
      });
    } else if (port.get_opcode() == LIBC_WRITE_TO_STDERR) {
      std::fill(files, files + num_lanes, stderr);
    } else {
      std::fill(files, files + num_lanes, stdout);
    }

    port.recv_n(strs, sizes,
                [&](uint64_t size) { return temp_storage.alloc(size); });
    port.send([&](rpc::Buffer *buffer, uint32_t id) {
      flockfile(files[id]);
      buffer->data[0] = fwrite_unlocked(strs[id], 1, sizes[id], files[id]);
      if (port.get_opcode() == LIBC_WRITE_TO_STDOUT_NEWLINE &&
          buffer->data[0] == sizes[id])
        buffer->data[0] += fwrite_unlocked("\n", 1, 1, files[id]);
      funlockfile(files[id]);
    });
    break;
  }
  case LIBC_READ_FROM_STREAM: {
    uint64_t sizes[num_lanes] = {0};
    void *data[num_lanes] = {nullptr};
    port.recv([&](rpc::Buffer *buffer, uint32_t id) {
      data[id] = temp_storage.alloc(buffer->data[0]);
      sizes[id] =
          fread(data[id], 1, buffer->data[0], to_stream(buffer->data[1]));
    });
    port.send_n(data, sizes);
    port.send([&](rpc::Buffer *buffer, uint32_t id) {
      std::memcpy(buffer->data, &sizes[id], sizeof(uint64_t));
    });
    break;
  }
  case LIBC_READ_FGETS: {
    uint64_t sizes[num_lanes] = {0};
    void *data[num_lanes] = {nullptr};
    port.recv([&](rpc::Buffer *buffer, uint32_t id) {
      data[id] = temp_storage.alloc(buffer->data[0]);
      const char *str = fgets(reinterpret_cast<char *>(data[id]),
                              buffer->data[0], to_stream(buffer->data[1]));
      sizes[id] = !str ? 0 : std::strlen(str) + 1;
    });
    port.send_n(data, sizes);
    break;
  }
  case LIBC_OPEN_FILE: {
    uint64_t sizes[num_lanes] = {0};
    void *paths[num_lanes] = {nullptr};
    port.recv_n(paths, sizes,
                [&](uint64_t size) { return temp_storage.alloc(size); });
    port.recv_and_send([&](rpc::Buffer *buffer, uint32_t id) {
      FILE *file = fopen(reinterpret_cast<char *>(paths[id]),
                         reinterpret_cast<char *>(buffer->data));
      buffer->data[0] = reinterpret_cast<uintptr_t>(file);
    });
    break;
  }
  case LIBC_CLOSE_FILE: {
    port.recv_and_send([&](rpc::Buffer *buffer, uint32_t id) {
      FILE *file = reinterpret_cast<FILE *>(buffer->data[0]);
      buffer->data[0] = fclose(file);
    });
    break;
  }
  case LIBC_EXIT: {
    // Send a response to the client to signal that we are ready to exit.
    port.recv_and_send([](rpc::Buffer *, uint32_t) {});
    port.recv([](rpc::Buffer *buffer, uint32_t) {
      int status = 0;
      std::memcpy(&status, buffer->data, sizeof(int));
      exit(status);
    });
    break;
  }
  case LIBC_ABORT: {
    // Send a response to the client to signal that we are ready to abort.
    port.recv_and_send([](rpc::Buffer *, uint32_t) {});
    port.recv([](rpc::Buffer *, uint32_t) {});
    abort();
    break;
  }
  case LIBC_HOST_CALL: {
    uint64_t sizes[num_lanes] = {0};
    unsigned long long results[num_lanes] = {0};
    void *args[num_lanes] = {nullptr};
    port.recv_n(args, sizes,
                [&](uint64_t size) { return temp_storage.alloc(size); });
    port.recv([&](rpc::Buffer *buffer, uint32_t id) {
      using func_ptr_t = unsigned long long (*)(void *);
      auto func = reinterpret_cast<func_ptr_t>(buffer->data[0]);
      results[id] = func(args[id]);
    });
    port.send([&](rpc::Buffer *buffer, uint32_t id) {
      buffer->data[0] = static_cast<uint64_t>(results[id]);
    });
    break;
  }
  case LIBC_FEOF: {
    port.recv_and_send([](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] = feof(to_stream(buffer->data[0]));
    });
    break;
  }
  case LIBC_FERROR: {
    port.recv_and_send([](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] = ferror(to_stream(buffer->data[0]));
    });
    break;
  }
  case LIBC_CLEARERR: {
    port.recv_and_send([](rpc::Buffer *buffer, uint32_t) {
      clearerr(to_stream(buffer->data[0]));
    });
    break;
  }
  case LIBC_FSEEK: {
    port.recv_and_send([](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] =
          fseek(to_stream(buffer->data[0]), static_cast<long>(buffer->data[1]),
                static_cast<int>(buffer->data[2]));
    });
    break;
  }
  case LIBC_FTELL: {
    port.recv_and_send([](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] = ftell(to_stream(buffer->data[0]));
    });
    break;
  }
  case LIBC_FFLUSH: {
    port.recv_and_send([](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] = fflush(to_stream(buffer->data[0]));
    });
    break;
  }
  case LIBC_UNGETC: {
    port.recv_and_send([](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] =
          ungetc(static_cast<int>(buffer->data[0]), to_stream(buffer->data[1]));
    });
    break;
  }
  case LIBC_PRINTF_TO_STREAM_PACKED:
  case LIBC_PRINTF_TO_STDOUT_PACKED:
  case LIBC_PRINTF_TO_STDERR_PACKED: {
    handle_printf<true, num_lanes>(port, temp_storage);
    break;
  }
  case LIBC_PRINTF_TO_STREAM:
  case LIBC_PRINTF_TO_STDOUT:
  case LIBC_PRINTF_TO_STDERR: {
    handle_printf<false, num_lanes>(port, temp_storage);
    break;
  }
  case LIBC_REMOVE: {
    uint64_t sizes[num_lanes] = {0};
    void *args[num_lanes] = {nullptr};
    port.recv_n(args, sizes,
                [&](uint64_t size) { return temp_storage.alloc(size); });
    port.send([&](rpc::Buffer *buffer, uint32_t id) {
      buffer->data[0] = static_cast<uint64_t>(
          remove(reinterpret_cast<const char *>(args[id])));
    });
    break;
  }
  case LIBC_RENAME: {
    uint64_t oldsizes[num_lanes] = {0};
    uint64_t newsizes[num_lanes] = {0};
    void *oldpath[num_lanes] = {nullptr};
    void *newpath[num_lanes] = {nullptr};
    port.recv_n(oldpath, oldsizes,
                [&](uint64_t size) { return temp_storage.alloc(size); });
    port.recv_n(newpath, newsizes,
                [&](uint64_t size) { return temp_storage.alloc(size); });
    port.send([&](rpc::Buffer *buffer, uint32_t id) {
      buffer->data[0] = static_cast<uint64_t>(
          rename(reinterpret_cast<const char *>(oldpath[id]),
                 reinterpret_cast<const char *>(newpath[id])));
    });
    break;
  }
  case LIBC_SYSTEM: {
    uint64_t sizes[num_lanes] = {0};
    void *args[num_lanes] = {nullptr};
    port.recv_n(args, sizes,
                [&](uint64_t size) { return temp_storage.alloc(size); });
    port.send([&](rpc::Buffer *buffer, uint32_t id) {
      buffer->data[0] = static_cast<uint64_t>(
          system(reinterpret_cast<const char *>(args[id])));
    });
    break;
  }
  case LIBC_NOOP: {
    port.recv([](rpc::Buffer *, uint32_t) {});
    break;
  }
  default:
    return rpc::RPC_UNHANDLED_OPCODE;
  }

  return rpc::RPC_SUCCESS;
}

namespace rpc {
// The implementation of this function currently lives in the utility directory
// at 'utils/gpu/server/rpc_server.cpp'.
rpc::Status handle_libc_opcodes(rpc::Server::Port &port, uint32_t num_lanes) {
  switch (num_lanes) {
  case 1:
    return handle_port_impl<1>(port);
  case 32:
    return handle_port_impl<32>(port);
  case 64:
    return handle_port_impl<64>(port);
  default:
    return rpc::RPC_ERROR;
  }
}
} // namespace rpc
