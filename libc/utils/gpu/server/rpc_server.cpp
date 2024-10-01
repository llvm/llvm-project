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

#include "llvmlibc_rpc_server.h"

#include "src/__support/RPC/rpc.h"
#include "src/__support/arg_list.h"
#include "src/stdio/printf_core/converter.h"
#include "src/stdio/printf_core/parser.h"
#include "src/stdio/printf_core/writer.h"

#include "src/stdio/gpu/file.h"
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

static_assert(sizeof(rpc_buffer_t) == sizeof(rpc::Buffer),
              "Buffer size mismatch");

static_assert(RPC_MAXIMUM_PORT_COUNT == rpc::MAX_PORT_COUNT,
              "Incorrect maximum port count");

namespace {
struct TempStorage {
  char *alloc(size_t size) {
    storage.emplace_back(std::make_unique<char[]>(size));
    return storage.back().get();
  }

  std::vector<std::unique_ptr<char[]>> storage;
};
} // namespace

template <bool packed, uint32_t lane_size>
static void handle_printf(rpc::Server::Port &port, TempStorage &temp_storage) {
  FILE *files[lane_size] = {nullptr};
  // Get the appropriate output stream to use.
  if (port.get_opcode() == RPC_PRINTF_TO_STREAM ||
      port.get_opcode() == RPC_PRINTF_TO_STREAM_PACKED)
    port.recv([&](rpc::Buffer *buffer, uint32_t id) {
      files[id] = reinterpret_cast<FILE *>(buffer->data[0]);
    });
  else if (port.get_opcode() == RPC_PRINTF_TO_STDOUT ||
           port.get_opcode() == RPC_PRINTF_TO_STDOUT_PACKED)
    std::fill(files, files + lane_size, stdout);
  else
    std::fill(files, files + lane_size, stderr);

  uint64_t format_sizes[lane_size] = {0};
  void *format[lane_size] = {nullptr};

  uint64_t args_sizes[lane_size] = {0};
  void *args[lane_size] = {nullptr};

  // Recieve the format string and arguments from the client.
  port.recv_n(format, format_sizes,
              [&](uint64_t size) { return temp_storage.alloc(size); });

  // Parse the format string to get the expected size of the buffer.
  for (uint32_t lane = 0; lane < lane_size; ++lane) {
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
  std::vector<void *> strs_to_copy[lane_size];
  int buffer_size[lane_size] = {0};
  for (uint32_t lane = 0; lane < lane_size; ++lane) {
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
  std::vector<void *> copied_strs[lane_size];
  while (std::any_of(std::begin(strs_to_copy), std::end(strs_to_copy),
                     [](const auto &v) { return !v.empty() && v.back(); })) {
    port.send([&](rpc::Buffer *buffer, uint32_t id) {
      void *ptr = !strs_to_copy[id].empty() ? strs_to_copy[id].back() : nullptr;
      buffer->data[1] = reinterpret_cast<uintptr_t>(ptr);
      if (!strs_to_copy[id].empty())
        strs_to_copy[id].pop_back();
    });
    uint64_t str_sizes[lane_size] = {0};
    void *strs[lane_size] = {nullptr};
    port.recv_n(strs, str_sizes,
                [&](uint64_t size) { return temp_storage.alloc(size); });
    for (uint32_t lane = 0; lane < lane_size; ++lane) {
      if (!strs[lane])
        continue;

      copied_strs[lane].emplace_back(strs[lane]);
      buffer_size[lane] += str_sizes[lane];
    }
  }

  // Perform the final formatting and printing using the LLVM C library printf.
  int results[lane_size] = {0};
  for (uint32_t lane = 0; lane < lane_size; ++lane) {
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

template <uint32_t lane_size>
rpc_status_t handle_server_impl(
    rpc::Server &server,
    const std::unordered_map<uint16_t, rpc_opcode_callback_ty> &callbacks,
    const std::unordered_map<uint16_t, void *> &callback_data,
    uint32_t &index) {
  auto port = server.try_open(lane_size, index);
  if (!port)
    return RPC_STATUS_SUCCESS;

  TempStorage temp_storage;

  switch (port->get_opcode()) {
  case RPC_WRITE_TO_STREAM:
  case RPC_WRITE_TO_STDERR:
  case RPC_WRITE_TO_STDOUT:
  case RPC_WRITE_TO_STDOUT_NEWLINE: {
    uint64_t sizes[lane_size] = {0};
    void *strs[lane_size] = {nullptr};
    FILE *files[lane_size] = {nullptr};
    if (port->get_opcode() == RPC_WRITE_TO_STREAM) {
      port->recv([&](rpc::Buffer *buffer, uint32_t id) {
        files[id] = reinterpret_cast<FILE *>(buffer->data[0]);
      });
    } else if (port->get_opcode() == RPC_WRITE_TO_STDERR) {
      std::fill(files, files + lane_size, stderr);
    } else {
      std::fill(files, files + lane_size, stdout);
    }

    port->recv_n(strs, sizes,
                 [&](uint64_t size) { return temp_storage.alloc(size); });
    port->send([&](rpc::Buffer *buffer, uint32_t id) {
      flockfile(files[id]);
      buffer->data[0] = fwrite_unlocked(strs[id], 1, sizes[id], files[id]);
      if (port->get_opcode() == RPC_WRITE_TO_STDOUT_NEWLINE &&
          buffer->data[0] == sizes[id])
        buffer->data[0] += fwrite_unlocked("\n", 1, 1, files[id]);
      funlockfile(files[id]);
    });
    break;
  }
  case RPC_READ_FROM_STREAM: {
    uint64_t sizes[lane_size] = {0};
    void *data[lane_size] = {nullptr};
    port->recv([&](rpc::Buffer *buffer, uint32_t id) {
      data[id] = temp_storage.alloc(buffer->data[0]);
      sizes[id] =
          fread(data[id], 1, buffer->data[0], file::to_stream(buffer->data[1]));
    });
    port->send_n(data, sizes);
    port->send([&](rpc::Buffer *buffer, uint32_t id) {
      std::memcpy(buffer->data, &sizes[id], sizeof(uint64_t));
    });
    break;
  }
  case RPC_READ_FGETS: {
    uint64_t sizes[lane_size] = {0};
    void *data[lane_size] = {nullptr};
    port->recv([&](rpc::Buffer *buffer, uint32_t id) {
      data[id] = temp_storage.alloc(buffer->data[0]);
      const char *str =
          fgets(reinterpret_cast<char *>(data[id]), buffer->data[0],
                file::to_stream(buffer->data[1]));
      sizes[id] = !str ? 0 : std::strlen(str) + 1;
    });
    port->send_n(data, sizes);
    break;
  }
  case RPC_OPEN_FILE: {
    uint64_t sizes[lane_size] = {0};
    void *paths[lane_size] = {nullptr};
    port->recv_n(paths, sizes,
                 [&](uint64_t size) { return temp_storage.alloc(size); });
    port->recv_and_send([&](rpc::Buffer *buffer, uint32_t id) {
      FILE *file = fopen(reinterpret_cast<char *>(paths[id]),
                         reinterpret_cast<char *>(buffer->data));
      buffer->data[0] = reinterpret_cast<uintptr_t>(file);
    });
    break;
  }
  case RPC_CLOSE_FILE: {
    port->recv_and_send([&](rpc::Buffer *buffer, uint32_t id) {
      FILE *file = reinterpret_cast<FILE *>(buffer->data[0]);
      buffer->data[0] = fclose(file);
    });
    break;
  }
  case RPC_EXIT: {
    // Send a response to the client to signal that we are ready to exit.
    port->recv_and_send([](rpc::Buffer *) {});
    port->recv([](rpc::Buffer *buffer) {
      int status = 0;
      std::memcpy(&status, buffer->data, sizeof(int));
      exit(status);
    });
    break;
  }
  case RPC_ABORT: {
    // Send a response to the client to signal that we are ready to abort.
    port->recv_and_send([](rpc::Buffer *) {});
    port->recv([](rpc::Buffer *) {});
    abort();
    break;
  }
  case RPC_HOST_CALL: {
    uint64_t sizes[lane_size] = {0};
    void *args[lane_size] = {nullptr};
    port->recv_n(args, sizes,
                 [&](uint64_t size) { return temp_storage.alloc(size); });
    port->recv([&](rpc::Buffer *buffer, uint32_t id) {
      reinterpret_cast<void (*)(void *)>(buffer->data[0])(args[id]);
    });
    port->send([&](rpc::Buffer *, uint32_t id) {});
    break;
  }
  case RPC_FEOF: {
    port->recv_and_send([](rpc::Buffer *buffer) {
      buffer->data[0] = feof(file::to_stream(buffer->data[0]));
    });
    break;
  }
  case RPC_FERROR: {
    port->recv_and_send([](rpc::Buffer *buffer) {
      buffer->data[0] = ferror(file::to_stream(buffer->data[0]));
    });
    break;
  }
  case RPC_CLEARERR: {
    port->recv_and_send([](rpc::Buffer *buffer) {
      clearerr(file::to_stream(buffer->data[0]));
    });
    break;
  }
  case RPC_FSEEK: {
    port->recv_and_send([](rpc::Buffer *buffer) {
      buffer->data[0] = fseek(file::to_stream(buffer->data[0]),
                              static_cast<long>(buffer->data[1]),
                              static_cast<int>(buffer->data[2]));
    });
    break;
  }
  case RPC_FTELL: {
    port->recv_and_send([](rpc::Buffer *buffer) {
      buffer->data[0] = ftell(file::to_stream(buffer->data[0]));
    });
    break;
  }
  case RPC_FFLUSH: {
    port->recv_and_send([](rpc::Buffer *buffer) {
      buffer->data[0] = fflush(file::to_stream(buffer->data[0]));
    });
    break;
  }
  case RPC_UNGETC: {
    port->recv_and_send([](rpc::Buffer *buffer) {
      buffer->data[0] = ungetc(static_cast<int>(buffer->data[0]),
                               file::to_stream(buffer->data[1]));
    });
    break;
  }
  case RPC_PRINTF_TO_STREAM_PACKED:
  case RPC_PRINTF_TO_STDOUT_PACKED:
  case RPC_PRINTF_TO_STDERR_PACKED: {
    handle_printf<true, lane_size>(*port, temp_storage);
    break;
  }
  case RPC_PRINTF_TO_STREAM:
  case RPC_PRINTF_TO_STDOUT:
  case RPC_PRINTF_TO_STDERR: {
    handle_printf<false, lane_size>(*port, temp_storage);
    break;
  }
  case RPC_REMOVE: {
    uint64_t sizes[lane_size] = {0};
    void *args[lane_size] = {nullptr};
    port->recv_n(args, sizes,
                 [&](uint64_t size) { return temp_storage.alloc(size); });
    port->send([&](rpc::Buffer *buffer, uint32_t id) {
      buffer->data[0] = static_cast<uint64_t>(
          remove(reinterpret_cast<const char *>(args[id])));
    });
    break;
  }
  case RPC_RENAME: {
    uint64_t oldsizes[lane_size] = {0};
    uint64_t newsizes[lane_size] = {0};
    void *oldpath[lane_size] = {nullptr};
    void *newpath[lane_size] = {nullptr};
    port->recv_n(oldpath, oldsizes,
                 [&](uint64_t size) { return temp_storage.alloc(size); });
    port->recv_n(newpath, newsizes,
                 [&](uint64_t size) { return temp_storage.alloc(size); });
    port->send([&](rpc::Buffer *buffer, uint32_t id) {
      buffer->data[0] = static_cast<uint64_t>(
          rename(reinterpret_cast<const char *>(oldpath[id]),
                 reinterpret_cast<const char *>(newpath[id])));
    });
    break;
  }
  case RPC_SYSTEM: {
    uint64_t sizes[lane_size] = {0};
    void *args[lane_size] = {nullptr};
    port->recv_n(args, sizes,
                 [&](uint64_t size) { return temp_storage.alloc(size); });
    port->send([&](rpc::Buffer *buffer, uint32_t id) {
      buffer->data[0] = static_cast<uint64_t>(
          system(reinterpret_cast<const char *>(args[id])));
    });
    break;
  }
  case RPC_NOOP: {
    port->recv([](rpc::Buffer *) {});
    break;
  }
  default: {
    auto handler =
        callbacks.find(static_cast<rpc_opcode_t>(port->get_opcode()));

    // We error out on an unhandled opcode.
    if (handler == callbacks.end())
      return RPC_STATUS_UNHANDLED_OPCODE;

    // Invoke the registered callback with a reference to the port.
    void *data =
        callback_data.at(static_cast<rpc_opcode_t>(port->get_opcode()));
    rpc_port_t port_ref{reinterpret_cast<uint64_t>(&*port), lane_size};
    (handler->second)(port_ref, data);
  }
  }

  // Increment the index so we start the scan after this port.
  index = port->get_index() + 1;
  port->close();

  return RPC_STATUS_CONTINUE;
}

struct Device {
  Device(uint32_t lane_size, uint32_t num_ports, void *buffer)
      : lane_size(lane_size), buffer(buffer), server(num_ports, buffer),
        client(num_ports, buffer) {}

  rpc_status_t handle_server(uint32_t &index) {
    switch (lane_size) {
    case 1:
      return handle_server_impl<1>(server, callbacks, callback_data, index);
    case 32:
      return handle_server_impl<32>(server, callbacks, callback_data, index);
    case 64:
      return handle_server_impl<64>(server, callbacks, callback_data, index);
    default:
      return RPC_STATUS_INVALID_LANE_SIZE;
    }
  }

  uint32_t lane_size;
  void *buffer;
  rpc::Server server;
  rpc::Client client;
  std::unordered_map<uint16_t, rpc_opcode_callback_ty> callbacks;
  std::unordered_map<uint16_t, void *> callback_data;
};

rpc_status_t rpc_server_init(rpc_device_t *rpc_device, uint64_t num_ports,
                             uint32_t lane_size, rpc_alloc_ty alloc,
                             void *data) {
  if (!rpc_device)
    return RPC_STATUS_ERROR;
  if (lane_size != 1 && lane_size != 32 && lane_size != 64)
    return RPC_STATUS_INVALID_LANE_SIZE;

  uint64_t size = rpc::Server::allocation_size(lane_size, num_ports);
  void *buffer = alloc(size, data);

  if (!buffer)
    return RPC_STATUS_ERROR;

  Device *device = new Device(lane_size, num_ports, buffer);
  if (!device)
    return RPC_STATUS_ERROR;

  rpc_device->handle = reinterpret_cast<uintptr_t>(device);
  return RPC_STATUS_SUCCESS;
}

rpc_status_t rpc_server_shutdown(rpc_device_t rpc_device, rpc_free_ty dealloc,
                                 void *data) {
  if (!rpc_device.handle)
    return RPC_STATUS_ERROR;

  Device *device = reinterpret_cast<Device *>(rpc_device.handle);
  dealloc(device->buffer, data);
  delete device;

  return RPC_STATUS_SUCCESS;
}

rpc_status_t rpc_handle_server(rpc_device_t rpc_device) {
  if (!rpc_device.handle)
    return RPC_STATUS_ERROR;

  Device *device = reinterpret_cast<Device *>(rpc_device.handle);
  uint32_t index = 0;
  for (;;) {
    rpc_status_t status = device->handle_server(index);
    if (status != RPC_STATUS_CONTINUE)
      return status;
  }
}

rpc_status_t rpc_register_callback(rpc_device_t rpc_device, uint16_t opcode,
                                   rpc_opcode_callback_ty callback,
                                   void *data) {
  if (!rpc_device.handle)
    return RPC_STATUS_ERROR;

  Device *device = reinterpret_cast<Device *>(rpc_device.handle);

  device->callbacks[opcode] = callback;
  device->callback_data[opcode] = data;
  return RPC_STATUS_SUCCESS;
}

const void *rpc_get_client_buffer(rpc_device_t rpc_device) {
  if (!rpc_device.handle)
    return nullptr;
  Device *device = reinterpret_cast<Device *>(rpc_device.handle);
  return &device->client;
}

uint64_t rpc_get_client_size() { return sizeof(rpc::Client); }

void rpc_send(rpc_port_t ref, rpc_port_callback_ty callback, void *data) {
  auto port = reinterpret_cast<rpc::Server::Port *>(ref.handle);
  port->send([=](rpc::Buffer *buffer) {
    callback(reinterpret_cast<rpc_buffer_t *>(buffer), data);
  });
}

void rpc_send_n(rpc_port_t ref, const void *const *src, uint64_t *size) {
  auto port = reinterpret_cast<rpc::Server::Port *>(ref.handle);
  port->send_n(src, size);
}

void rpc_recv(rpc_port_t ref, rpc_port_callback_ty callback, void *data) {
  auto port = reinterpret_cast<rpc::Server::Port *>(ref.handle);
  port->recv([=](rpc::Buffer *buffer) {
    callback(reinterpret_cast<rpc_buffer_t *>(buffer), data);
  });
}

void rpc_recv_n(rpc_port_t ref, void **dst, uint64_t *size, rpc_alloc_ty alloc,
                void *data) {
  auto port = reinterpret_cast<rpc::Server::Port *>(ref.handle);
  auto alloc_fn = [=](uint64_t size) { return alloc(size, data); };
  port->recv_n(dst, size, alloc_fn);
}

void rpc_recv_and_send(rpc_port_t ref, rpc_port_callback_ty callback,
                       void *data) {
  auto port = reinterpret_cast<rpc::Server::Port *>(ref.handle);
  port->recv_and_send([=](rpc::Buffer *buffer) {
    callback(reinterpret_cast<rpc_buffer_t *>(buffer), data);
  });
}
