// RUN: %clang_cc1 -emit-llvm -triple=x86_64-pc-win32 -fms-compatibility %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fms-compatibility -emit-pch -o %t %s
// RUN: %clang_cc1 -emit-llvm -triple=x86_64-pc-win32 -fms-compatibility -include-pch %t -verify %s -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

struct S {
  int GetX() { return 42; }
  __declspec(property(get=GetX)) int x;

  int GetY(int i, int j) { return i+j; }
  __declspec(property(get=GetY)) int y[];

  void* operator new(__SIZE_TYPE__, int);
};

template <typename T>
struct TS {
  T GetT() { return T(); }
  __declspec(property(get=GetT)) T t;

  T GetR(T i, T j) { return i+j; }
  __declspec(property(get=GetR)) T r[];
};

// CHECK-LABEL: main
int main(int argc, char **argv) {
  S *s;
  TS<double> *ts;

  // CHECK: [[X:%.+]] = call noundef i32 @"?GetX@S@@QEAAHXZ"(ptr {{[^,]*}} %{{.+}})
  // CHECK-NEXT: call noundef ptr @"??2S@@SAPEAX_KH@Z"(i64 noundef 1, i32 noundef [[X]])
  new (s->x) S;

  // CHECK: [[Y:%.+]] = call noundef i32 @"?GetY@S@@QEAAHHH@Z"(ptr {{[^,]*}} %{{.+}}, i32 noundef 1, i32 noundef 2)
  // CHECK-NEXT: call noundef ptr @"??2S@@SAPEAX_KH@Z"(i64 noundef 1, i32 noundef [[Y]])
  new ((s->y)[1][2]) S;

  // CHECK: [[T:%.+]] = call noundef double @"?GetT@?$TS@N@@QEAANXZ"(ptr {{[^,]*}} %{{.+}})
  // CHECK-NEXT: [[TI:%.+]] = fptosi double [[T]] to i32
  // CHECK-NEXT: call noundef ptr @"??2S@@SAPEAX_KH@Z"(i64 noundef 1, i32 noundef [[TI]])
  new (ts->t) S;

  // CHECK: [[R:%.+]] = call noundef double @"?GetR@?$TS@N@@QEAANNN@Z"(ptr {{[^,]*}} %{{.+}}, double {{[^,]*}}, double {{[^,]*}})
  // CHECK-NEXT: [[RI:%.+]] = fptosi double [[R]] to i32
  // CHECK-NEXT: call noundef ptr @"??2S@@SAPEAX_KH@Z"(i64 noundef 1, i32 noundef [[RI]])
  new ((ts->r)[1][2]) S;
}

#endif
