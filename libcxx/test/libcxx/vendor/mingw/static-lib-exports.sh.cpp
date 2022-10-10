//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: target={{.+}}-windows-gnu

// This file checks that the built static libraries don't contain dllexport
// directives in MinGW builds.

// RUN: llvm-readobj --coff-directives "%{lib}/libc++.a" | not grep -i "export:" > /dev/null

// RUN: llvm-readobj --coff-directives "%{lib}/libc++experimental.a" | not grep -i "export:" > /dev/null
