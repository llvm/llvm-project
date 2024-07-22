// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -std=c++11 -emit-llvm -o - %s | FileCheck %s

struct S {
  void* operator new(__SIZE_TYPE__, int);
};

int main() {
  // CHECK: call {{.*}} ptr @"??2S@@SAPEAX_KH@Z"(i64 {{.*}} 1, i32 {{.*}} 0)
  // CHECK: call {{.*}} ptr @"??2S@@SAPEAX_KH@Z"(i64 {{.*}} 1, i32 {{.*}} 0)
  new (__noop) S;
  new ((__noop)) S;
}
