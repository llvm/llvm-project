//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// REQUIRES: host-has-gdb-with-python

// Android org doesn't support GDB anymore.
// UNSUPPORTED: android

// LeakSanitizer does not work under ptrace
// UNSUPPORTED: asan

// GDB doesn't support PDB debug info format
// UNSUPPORTED: msvc

// Installed GDB on windows-on-arm is x86_64 only
// UNSUPPORTED: target=aarch64-w64-windows-gnu

// XFAIL: LIBCXX-PICOLIBC-FIXME

// RUN: %{cxx} %{flags} %s -o %t.exe %{compile_flags} -g %{link_flags}
// RUN: %{exec} %{gdb} --return-child-result -ex run %t.exe

// <debugging>

// bool is_debugger_present() noexcept;

#include <debugging>

int main(int, char**) { return std::is_debugger_present() ? 0 : 1; }
