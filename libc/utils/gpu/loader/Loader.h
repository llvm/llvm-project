//===-- Generic device loader interface -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stddef.h>

/// Generic interface to load the \p image and launch execution of the _start
/// kernel on the target device. Copies \p argc and \p argv to the device.
/// Returns the final value of the `main` function on the device.
int load(int argc, char **argv, void *image, size_t size);
