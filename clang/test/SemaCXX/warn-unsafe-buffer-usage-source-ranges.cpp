// RUN: %clang_cc1 -Wno-everything -Wunsafe-buffer-usage -fdiagnostics-print-source-range-info %s 2>&1 | FileCheck %s

void foo(int i) {
  int * ptr;

  ptr++;
  // CHECK-DAG: {[[@LINE-1]]:3-[[@LINE-1]]:6}
  ptr--;
  // CHECK-DAG: {[[@LINE-1]]:3-[[@LINE-1]]:6}
  ++ptr;
  // CHECK-DAG: {[[@LINE-1]]:5-[[@LINE-1]]:8}
  --ptr;
  // CHECK-DAG: {[[@LINE-1]]:5-[[@LINE-1]]:8}


  ptr + 1;
  // CHECK-DAG: {[[@LINE-1]]:3-[[@LINE-1]]:6}
  2 + ptr;
  // CHECK-DAG: {[[@LINE-1]]:7-[[@LINE-1]]:10}
  ptr + i;
  // CHECK-DAG: {[[@LINE-1]]:3-[[@LINE-1]]:6}
  i + ptr;
  // CHECK-DAG: {[[@LINE-1]]:7-[[@LINE-1]]:10}


  ptr - 3;
  // CHECK-DAG: {[[@LINE-1]]:3-[[@LINE-1]]:6}
  ptr - i;
  // CHECK-DAG: {[[@LINE-1]]:3-[[@LINE-1]]:6}


  ptr += 4;
  // CHECK-DAG: {[[@LINE-1]]:3-[[@LINE-1]]:6}
  ptr += i;
  // CHECK-DAG: {[[@LINE-1]]:3-[[@LINE-1]]:6}


  ptr -= 5;
  // CHECK-DAG: {[[@LINE-1]]:3-[[@LINE-1]]:6}
  ptr -= i;
  // CHECK-DAG: {[[@LINE-1]]:3-[[@LINE-1]]:6}


  ptr[5];
  // CHECK-DAG: {[[@LINE-1]]:3-[[@LINE-1]]:6}
  5[ptr];
  // CHECK-DAG: {[[@LINE-1]]:5-[[@LINE-1]]:8}
}
