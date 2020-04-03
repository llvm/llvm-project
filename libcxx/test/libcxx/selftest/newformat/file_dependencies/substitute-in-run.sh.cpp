//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure %{file_dependencies} are substituted properly into RUN commands.

// FILE_DEPENDENCIES: a.txt
// FILE_DEPENDENCIES: b.txt, c.txt
// FILE_DEPENDENCIES: /absolute/d.txt
// RUN: echo %{file_dependencies} | grep 'a.txt'
// RUN: echo %{file_dependencies} | grep 'b.txt'
// RUN: echo %{file_dependencies} | grep 'c.txt'
// RUN: echo %{file_dependencies} | grep '/absolute/d.txt'
