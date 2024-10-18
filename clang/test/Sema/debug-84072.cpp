// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

void Func(int x) {
  switch (x) {
    [[likely]] case 0:
    case 1:
      int i = 3;
    case 2:
      break;
  }

//  switch (x) {
//    case 0:
//    case 1:
//      int i = 3;
//    case 2:
//      break;
//  }
}
