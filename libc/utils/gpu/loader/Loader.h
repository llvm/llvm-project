//===-- Generic device loader interface -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_GPU_LOADER_LOADER_H
#define LLVM_LIBC_UTILS_GPU_LOADER_LOADER_H

#include "utils/gpu/server/llvmlibc_rpc_server.h"

#include "include/llvm-libc-types/test_rpc_opcodes_t.h"
#include "llvm-libc-types/rpc_opcodes_t.h"

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
int load(int argc, char **argv, char **evnp, void *image, size_t size,
         const LaunchParameters &params);

/// Return \p V aligned "upwards" according to \p Align.
template <typename V, typename A> inline V align_up(V val, A align) {
  return ((val + V(align) - 1) / V(align)) * V(align);
}

/// Copy the system's argument vector to GPU memory allocated using \p alloc.
template <typename Allocator>
void *copy_argument_vector(int argc, char **argv, Allocator alloc) {
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
  reinterpret_cast<void **>(dev_argv)[argv_size] = nullptr;
  return dev_argv;
}

/// Copy the system's environment to GPU memory allocated using \p alloc.
template <typename Allocator>
void *copy_environment(char **envp, Allocator alloc) {
  int envc = 0;
  for (char **env = envp; *env != 0; ++env)
    ++envc;

  return copy_argument_vector(envc, envp, alloc);
}

inline void handle_error_impl(const char *file, int32_t line, const char *msg) {
  fprintf(stderr, "%s:%d:0: Error: %s\n", file, line, msg);
  exit(EXIT_FAILURE);
}

inline void handle_error_impl(const char *file, int32_t line,
                              rpc_status_t err) {
  fprintf(stderr, "%s:%d:0: Error: %d\n", file, line, err);
  exit(EXIT_FAILURE);
}
#define handle_error(X) handle_error_impl(__FILE__, __LINE__, X)

template <uint32_t lane_size>
inline void register_rpc_callbacks(rpc_device_t device) {
  static_assert(lane_size == 32 || lane_size == 64, "Invalid Lane size");
  // Register the ping test for the `libc` tests.
  rpc_register_callback(
      device, static_cast<rpc_opcode_t>(RPC_TEST_INCREMENT),
      [](rpc_port_t port, void *data) {
        rpc_recv_and_send(
            port,
            [](rpc_buffer_t *buffer, void *data) {
              reinterpret_cast<uint64_t *>(buffer->data)[0] += 1;
            },
            data);
      },
      nullptr);

  // Register the interface test callbacks.
  rpc_register_callback(
      device, static_cast<rpc_opcode_t>(RPC_TEST_INTERFACE),
      [](rpc_port_t port, void *data) {
        uint64_t cnt = 0;
        bool end_with_recv;
        rpc_recv(
            port,
            [](rpc_buffer_t *buffer, void *data) {
              *reinterpret_cast<bool *>(data) = buffer->data[0];
            },
            &end_with_recv);
        rpc_recv(
            port,
            [](rpc_buffer_t *buffer, void *data) {
              *reinterpret_cast<uint64_t *>(data) = buffer->data[0];
            },
            &cnt);
        rpc_send(
            port,
            [](rpc_buffer_t *buffer, void *data) {
              uint64_t &cnt = *reinterpret_cast<uint64_t *>(data);
              buffer->data[0] = cnt = cnt + 1;
            },
            &cnt);
        rpc_recv(
            port,
            [](rpc_buffer_t *buffer, void *data) {
              *reinterpret_cast<uint64_t *>(data) = buffer->data[0];
            },
            &cnt);
        rpc_send(
            port,
            [](rpc_buffer_t *buffer, void *data) {
              uint64_t &cnt = *reinterpret_cast<uint64_t *>(data);
              buffer->data[0] = cnt = cnt + 1;
            },
            &cnt);
        rpc_recv(
            port,
            [](rpc_buffer_t *buffer, void *data) {
              *reinterpret_cast<uint64_t *>(data) = buffer->data[0];
            },
            &cnt);
        rpc_recv(
            port,
            [](rpc_buffer_t *buffer, void *data) {
              *reinterpret_cast<uint64_t *>(data) = buffer->data[0];
            },
            &cnt);
        rpc_send(
            port,
            [](rpc_buffer_t *buffer, void *data) {
              uint64_t &cnt = *reinterpret_cast<uint64_t *>(data);
              buffer->data[0] = cnt = cnt + 1;
            },
            &cnt);
        rpc_send(
            port,
            [](rpc_buffer_t *buffer, void *data) {
              uint64_t &cnt = *reinterpret_cast<uint64_t *>(data);
              buffer->data[0] = cnt = cnt + 1;
            },
            &cnt);
        if (end_with_recv)
          rpc_recv(
              port,
              [](rpc_buffer_t *buffer, void *data) {
                *reinterpret_cast<uint64_t *>(data) = buffer->data[0];
              },
              &cnt);
        else
          rpc_send(
              port,
              [](rpc_buffer_t *buffer, void *data) {
                uint64_t &cnt = *reinterpret_cast<uint64_t *>(data);
                buffer->data[0] = cnt = cnt + 1;
              },
              &cnt);
      },
      nullptr);

  // Register the stream test handler.
  rpc_register_callback(
      device, static_cast<rpc_opcode_t>(RPC_TEST_STREAM),
      [](rpc_port_t port, void *data) {
        uint64_t sizes[lane_size] = {0};
        void *dst[lane_size] = {nullptr};
        rpc_recv_n(
            port, dst, sizes,
            [](uint64_t size, void *) -> void * { return new char[size]; },
            nullptr);
        rpc_send_n(port, dst, sizes);
        for (uint64_t i = 0; i < lane_size; ++i) {
          if (dst[i])
            delete[] reinterpret_cast<uint8_t *>(dst[i]);
        }
      },
      nullptr);
}

#endif
