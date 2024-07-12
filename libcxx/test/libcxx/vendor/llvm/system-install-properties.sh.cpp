//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: stdlib=libc++ && !stdlib=apple-libc++

// DEFINE: %{shared-suffix}=%if target={{.+}}-apple-{{.+}} %{ dylib %} %else %{ so %}

// This file checks various properties of the installation of libc++ when built under
// a vanilla upstream configuration.

// TODO: temporary to debug
// RUN: ls %{lib-dir}

// Make sure we install the libc++ headers in the right location.
//
// RUN: stat "%{include-dir}/__config"

// Make sure we install libc++.1.dylib and libc++experimental.a in the right location.
//
// RUN: stat "%{lib-dir}/libc++.1.%{shared-suffix}"
// RUN: stat "%{lib-dir}/libc++experimental.a"

// Make sure we install a symlink from libc++.dylib to libc++.1.dylib.
//
// RUN: stat "%{lib-dir}/libc++.%{shared-suffix}"
// RUN: readlink "%{lib-dir}/libc++.%{shared-suffix}" | grep "libc++.1.%{shared-suffix}"
