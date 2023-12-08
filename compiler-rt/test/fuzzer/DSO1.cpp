// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source code for a simple DSO.
#ifdef _WIN32
__declspec( dllexport )
#endif
int DSO1(int a) {
  if (a < 123456)
    return 0;
  return 1;
}

void Uncovered1() { }
