//===-- Generic device loader interface -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_GPU_LOADER_LOADER_H
#define LLVM_LIBC_UTILS_GPU_LOADER_LOADER_H

#include <cstdint>
#include <cstring>
#include <stddef.h>

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
  void *inbox;
  void *outbox;
  void *buffer;
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
};

/// Copy the system's environment to GPU memory allocated using \p alloc.
template <typename Allocator>
void *copy_environment(char **envp, Allocator alloc) {
  int envc = 0;
  for (char **env = envp; *env != 0; ++env)
    ++envc;

  return copy_argument_vector(envc, envp, alloc);
};

#endif
