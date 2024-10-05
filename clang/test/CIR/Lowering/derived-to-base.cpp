// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

struct Base1 { int a; };
struct Base2 { int b; };
struct Derived : Base1, Base2 { int c; };
void test_multi_base() {
  Derived d;

  Base2& bref = d; // no null check needed
  // LLVM: %7 = getelementptr i8, ptr %1, i32 4

  Base2* bptr = &d; // has null pointer check
  // LLVM: %8 = icmp eq ptr %1, null
  // LLVM: %9 = getelementptr i8, ptr %1, i32 4
  // LLVM: %10 = select i1 %8, ptr %1, ptr %9

  int a = d.a;
  // LLVM: %11 = getelementptr i8, ptr %1, i32 0
  // LLVM: %12 = getelementptr %struct.Base1, ptr %11, i32 0, i32 0

  int b = d.b;
  // LLVM: %14 = getelementptr i8, ptr %1, i32 4
  // LLVM: %15 = getelementptr %struct.Base2, ptr %14, i32 0, i32 0

  int c = d.c;
  // LLVM: %17 = getelementptr %struct.Derived, ptr %1, i32 0, i32 2
}
