//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test requires access to the Swift compiler, which we assume is only available on Apple
// platforms for now.
// REQUIRES: buildhost=darwin

// std::string basic test
// RUN: swiftc -cxx-interoperability-mode=default %S/swift-interop/string/main.swift -o %t.string.exe \
// RUN:   -Xcc -nostdinc++ -Xcc -nostdlib++ -Xcc -isystem -Xcc "%{include-dir}"
// RUN: %{exec} %t.string.exe

// std::vector test with a modulemap
// RUN: swiftc -cxx-interoperability-mode=default %S/swift-interop/vector/main.swift -o %t.vector.exe \
// RUN:   -Xcc -nostdinc++ -Xcc -nostdlib++ -Xcc -isystem -Xcc "%{include-dir}" -Xcc -I -Xcc %S/swift-interop/vector
// RUN: %{exec} %t.vector.exe
