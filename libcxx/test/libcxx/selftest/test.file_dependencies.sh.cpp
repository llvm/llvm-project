// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FILE_DEPENDENCIES: test.pass.cpp
// FILE_DEPENDENCIES: %s

// RUN: echo %{file_dependencies} | grep 'test.pass.cpp'
// RUN: echo %{file_dependencies} | grep '%s'
