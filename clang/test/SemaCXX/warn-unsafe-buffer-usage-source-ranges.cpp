// RUN: %clang_cc1 -Wno-everything -Wunsafe-buffer-usage -fdiagnostics-print-source-range-info %s 2>&1 | FileCheck %s

void foo(int i) {
  int * ptr;


  ptr++;
  // CHECK-DAG: {7:3-7:6}{{.*}}[-Wunsafe-buffer-usage]
  ptr--;
  // CHECK-DAG: {9:3-9:6}{{.*}}[-Wunsafe-buffer-usage]
  ++ptr;
  // CHECK-DAG: {11:5-11:8}{{.*}}[-Wunsafe-buffer-usage]
  --ptr;
  // CHECK-DAG: {13:5-13:8}{{.*}}[-Wunsafe-buffer-usage]


  ptr + 1;
  // CHECK-DAG: {17:3-17:6}{{.*}}[-Wunsafe-buffer-usage]
  2 + ptr;
  // CHECK-DAG: {19:7-19:10}{{.*}}[-Wunsafe-buffer-usage]
  ptr + i;
  // CHECK-DAG: {21:3-21:6}{{.*}}[-Wunsafe-buffer-usage]
  i + ptr;
  // CHECK-DAG: {23:7-23:10}{{.*}}[-Wunsafe-buffer-usage]


  ptr - 3;
  // CHECK-DAG: {27:3-27:6}{{.*}}[-Wunsafe-buffer-usage]
  ptr - i;
  // CHECK-DAG: {29:3-29:6}{{.*}}[-Wunsafe-buffer-usage]


  ptr += 4;
  // CHECK-DAG: {33:3-33:6}{{.*}}[-Wunsafe-buffer-usage]
  ptr += i;
  // CHECK-DAG: {35:3-35:6}{{.*}}[-Wunsafe-buffer-usage]


  ptr -= 5;
  // CHECK-DAG: {39:3-39:6}{{.*}}[-Wunsafe-buffer-usage]
  ptr -= i;
  // CHECK-DAG: {41:3-41:6}{{.*}}[-Wunsafe-buffer-usage]


  ptr[5];
  // CHECK-DAG: {45:3-45:6}{{.*}}[-Wunsafe-buffer-usage]
  5[ptr];
  // CHECK-DAG: {47:5-47:8}{{.*}}[-Wunsafe-buffer-usage]
}
