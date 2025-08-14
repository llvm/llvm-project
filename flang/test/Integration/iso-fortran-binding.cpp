// REQUIRES: clang
// UNSUPPORTED: system-windows
// RUN: %clang_cc1 -fsyntax-only -I%flang_include %s -x c++

extern "C" {
#include "ISO_Fortran_binding.h"
}

int main() { return 0; }
