//===-- Generic device loader interface -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_GPU_LOADER_LOADER_H
#define LLVM_LIBC_UTILS_GPU_LOADER_LOADER_H

#include <cstring>
#include <stddef.h>

/// Generic interface to load the \p image and launch execution of the _start
/// kernel on the target device. Copies \p argc and \p argv to the device.
/// Returns the final value of the `main` function on the device.
int load(int argc, char **argv, char **evnp, void *image, size_t size);

/// Copy the system's argument vector to GPU memory allocated using \p alloc.
template <typename Allocator>
void *copy_argument_vector(int argc, char **argv, Allocator alloc) {
  void *dev_argv = alloc(argc * sizeof(char *));
  if (dev_argv == nullptr)
    return nullptr;

  for (int i = 0; i < argc; ++i) {
    size_t size = strlen(argv[i]) + 1;
    void *dev_str = alloc(size);
    if (dev_str == nullptr)
      return nullptr;

    // Load the host memory buffer with the pointer values of the newly
    // allocated strings.
    std::memcpy(dev_str, argv[i], size);
    static_cast<void **>(dev_argv)[i] = dev_str;
  }
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
