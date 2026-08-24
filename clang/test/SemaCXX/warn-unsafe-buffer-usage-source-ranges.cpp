// RUN: %clang_cc1 -Wno-everything -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-print-source-range-info %s 2>&1 | FileCheck %s

void foo(int i) {
  int * ptr;

  ptr++;
  // CHECK: {[[@LINE-1]]:3-[[@LINE-1]]:8}
  ptr--;
  // CHECK: {[[@LINE-1]]:3-[[@LINE-1]]:8}
  ++ptr;
  // CHECK: {[[@LINE-1]]:3-[[@LINE-1]]:8}
  --ptr;
  // CHECK: {[[@LINE-1]]:3-[[@LINE-1]]:8}


  ptr + 1;
  // CHECK: {[[@LINE-1]]:3-[[@LINE-1]]:10}
  2 + ptr;
  // CHECK: {[[@LINE-1]]:3-[[@LINE-1]]:10}
  ptr + i;
  // CHECK: {[[@LINE-1]]:3-[[@LINE-1]]:10}
  i + ptr;
  // CHECK: {[[@LINE-1]]:3-[[@LINE-1]]:10}


  ptr - 3;
  // CHECK: {[[@LINE-1]]:3-[[@LINE-1]]:10}
  ptr - i;
  // CHECK: {[[@LINE-1]]:3-[[@LINE-1]]:10}


  ptr += 4;
  // CHECK: {[[@LINE-1]]:3-[[@LINE-1]]:11}
  ptr += i;
  // CHECK: {[[@LINE-1]]:3-[[@LINE-1]]:11}


  ptr -= 5;
  // CHECK: {[[@LINE-1]]:3-[[@LINE-1]]:11}
  ptr -= i;
  // CHECK: {[[@LINE-1]]:3-[[@LINE-1]]:11}


  ptr[5];
  // CHECK: {[[@LINE-1]]:3-[[@LINE-1]]:9}
  5[ptr];
  // CHECK: {[[@LINE-1]]:3-[[@LINE-1]]:9}
}
