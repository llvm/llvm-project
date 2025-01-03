//===-- Generic device loader interface -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_GPU_LOADER_LOADER_H
#define LLVM_LIBC_UTILS_GPU_LOADER_LOADER_H

#include "include/llvm-libc-types/test_rpc_opcodes_t.h"

#include "shared/rpc.h"
#include "shared/rpc_opcodes.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/// Generic launch parameters for configuration the number of blocks / threads.
struct LaunchParameters {
  uint32_t num_threads_x;
  uint32_t num_threads_y;
  uint32_t num_threads_z;
  uint32_t num_blocks_x;
  uint32_t num_blocks_y;
  uint32_t num_blocks_z;
};

/// The arguments to the '_begin' kernel.
struct begin_args_t {
  int argc;
  void *argv;
  void *envp;
};

/// The arguments to the '_start' kernel.
struct start_args_t {
  int argc;
  void *argv;
  void *envp;
  void *ret;
};

/// The arguments to the '_end' kernel.
struct end_args_t {
  int argc;
};

/// Generic interface to load the \p image and launch execution of the _start
/// kernel on the target device. Copies \p argc and \p argv to the device.
/// Returns the final value of the `main` function on the device.
int load(int argc, const char **argv, const char **evnp, void *image,
         size_t size, const LaunchParameters &params,
         bool print_resource_usage);

/// Return \p V aligned "upwards" according to \p Align.
template <typename V, typename A> inline V align_up(V val, A align) {
  return ((val + V(align) - 1) / V(align)) * V(align);
}

/// Copy the system's argument vector to GPU memory allocated using \p alloc.
template <typename Allocator>
void *copy_argument_vector(int argc, const char **argv, Allocator alloc) {
  size_t argv_size = sizeof(char *) * (argc + 1);
  size_t str_size = 0;
  for (int i = 0; i < argc; ++i)
    str_size += strlen(argv[i]) + 1;

  // We allocate enough space for a null terminated array and all the strings.
  void *dev_argv = alloc(argv_size + str_size);
  if (!dev_argv)
    return nullptr;

  // Store the strings linerally in the same memory buffer.
  void *dev_str = reinterpret_cast<uint8_t *>(dev_argv) + argv_size;
  for (int i = 0; i < argc; ++i) {
    size_t size = strlen(argv[i]) + 1;
    std::memcpy(dev_str, argv[i], size);
    static_cast<void **>(dev_argv)[i] = dev_str;
    dev_str = reinterpret_cast<uint8_t *>(dev_str) + size;
  }

  // Ensure the vector is null terminated.
  reinterpret_cast<void **>(dev_argv)[argc] = nullptr;
  return dev_argv;
}

/// Copy the system's environment to GPU memory allocated using \p alloc.
template <typename Allocator>
void *copy_environment(const char **envp, Allocator alloc) {
  int envc = 0;
  for (const char **env = envp; *env != 0; ++env)
    ++envc;

  return copy_argument_vector(envc, envp, alloc);
}

inline void handle_error_impl(const char *file, int32_t line, const char *msg) {
  fprintf(stderr, "%s:%d:0: Error: %s\n", file, line, msg);
  exit(EXIT_FAILURE);
}
#define handle_error(X) handle_error_impl(__FILE__, __LINE__, X)

template <uint32_t num_lanes, typename Alloc, typename Free>
inline uint32_t handle_server(rpc::Server &server, uint32_t index,
                              Alloc &&alloc, Free &&free) {
  auto port = server.try_open(num_lanes, index);
  if (!port)
    return 0;
  index = port->get_index() + 1;

  int status = rpc::RPC_SUCCESS;
  switch (port->get_opcode()) {
  case RPC_TEST_INCREMENT: {
    port->recv_and_send([](rpc::Buffer *buffer, uint32_t) {
      reinterpret_cast<uint64_t *>(buffer->data)[0] += 1;
    });
    break;
  }
  case RPC_TEST_INTERFACE: {
    bool end_with_recv;
    uint64_t cnt;
    port->recv([&](rpc::Buffer *buffer, uint32_t) {
      end_with_recv = buffer->data[0];
    });
    port->recv([&](rpc::Buffer *buffer, uint32_t) { cnt = buffer->data[0]; });
    port->send([&](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] = cnt = cnt + 1;
    });
    port->recv([&](rpc::Buffer *buffer, uint32_t) { cnt = buffer->data[0]; });
    port->send([&](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] = cnt = cnt + 1;
    });
    port->recv([&](rpc::Buffer *buffer, uint32_t) { cnt = buffer->data[0]; });
    port->recv([&](rpc::Buffer *buffer, uint32_t) { cnt = buffer->data[0]; });
    port->send([&](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] = cnt = cnt + 1;
    });
    port->send([&](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] = cnt = cnt + 1;
    });
    if (end_with_recv)
      port->recv([&](rpc::Buffer *buffer, uint32_t) { cnt = buffer->data[0]; });
    else
      port->send([&](rpc::Buffer *buffer, uint32_t) {
        buffer->data[0] = cnt = cnt + 1;
      });

    break;
  }
  case RPC_TEST_STREAM: {
    uint64_t sizes[num_lanes] = {0};
    void *dst[num_lanes] = {nullptr};
    port->recv_n(dst, sizes,
                 [](uint64_t size) -> void * { return new char[size]; });
    port->send_n(dst, sizes);
    for (uint64_t i = 0; i < num_lanes; ++i) {
      if (dst[i])
        delete[] reinterpret_cast<uint8_t *>(dst[i]);
    }
    break;
  }
  case RPC_TEST_NOOP: {
    port->recv([&](rpc::Buffer *, uint32_t) {});
    break;
  }
  case LIBC_MALLOC: {
    port->recv_and_send([&](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] = reinterpret_cast<uintptr_t>(alloc(buffer->data[0]));
    });
    break;
  }
  case LIBC_FREE: {
    port->recv([&](rpc::Buffer *buffer, uint32_t) {
      free(reinterpret_cast<void *>(buffer->data[0]));
    });
    break;
  }
  default:
    status = handle_libc_opcodes(*port, num_lanes);
    break;
  }

  // Handle all of the `libc` specific opcodes.
  if (status != rpc::RPC_SUCCESS)
    handle_error("Error handling RPC server");

  port->close();

  return index;
}

#endif
