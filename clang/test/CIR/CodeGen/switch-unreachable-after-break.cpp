// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

void unreachable_after_break(int a) {
  switch(a) {
  case 0:
    break;
    break;
    int x = 1;
  }
}

int unreachable_after_return(int a) {
  switch (a) {
  case 0:
    return 0;
    return 1;
    int x = 1;
  }
  return 2;
}
