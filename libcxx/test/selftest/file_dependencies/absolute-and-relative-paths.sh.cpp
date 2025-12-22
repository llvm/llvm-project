//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that FILE_DEPENDENCIES work with relative AND absolute paths.

// FILE_DEPENDENCIES: %S/a.txt
// RUN: test -e %{temp}/a.txt

// FILE_DEPENDENCIES: dir/b.txt
// RUN: test -e %{temp}/b.txt
