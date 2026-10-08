//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-splitfile

// Make sure that we can use split-file to write tests when the `has-splitfile`
// Lit feature is defined.

// RUN: split-file %s %{temp}

// RUN: grep 'int main' %{temp}/main.cpp
// RUN: grep 'return 0' %{temp}/main.cpp
// RUN: not grep -c 'Pre-delimiter' %{temp}/main.cpp
// RUN: not grep -c 'foo' %{temp}/main.cpp
// RUN: not grep -c '//---' %{temp}/main.cpp

// RUN: grep foo %{temp}/input.txt
// RUN: grep bar %{temp}/input.txt
// RUN: not grep -c 'Pre-delimiter' %{temp}/input.txt
// RUN: not grep -c 'int main' %{temp}/input.txt
// RUN: not grep -c '//---' %{temp}/input.txt

// Pre-delimiter comment.

//--- main.cpp

int main() {
  return 0;
}

//--- input.txt

foo
bar
