// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

struct Base1 { int a; };
struct Base2 { int b; };
struct Derived : Base1, Base2 { int c; };
void test_multi_base() {
  Derived d;

  Base2& bref = d; // no null check needed
  // LLVM: getelementptr i8, ptr %[[D:.*]], i32 4

  Base2* bptr = &d; // has null pointer check
  // LLVM: %[[CHECK:.*]] = icmp eq ptr %[[D]], null
  // LLVM: %[[BPTR:.*]] = getelementptr i8, ptr %[[D]], i32 4
  // LLVM: select i1 %[[CHECK]], ptr %[[D]], ptr %[[BPTR]]

  int a = d.a;
  // LLVM: getelementptr %struct.Base1, ptr %[[D]], i32 0, i32 0

  int b = d.b;
  // LLVM: %[[BASE2_OFFSET:.*]] = getelementptr i8, ptr %[[D]], i32 4
  // LLVM: %[[BASE2:.*]] = getelementptr %struct.Base2, ptr %[[BASE2_OFFSET]], i32 0, i32 0

  int c = d.c;
  // LLVM: getelementptr %struct.Derived, ptr %[[D]], i32 0, i32 2
}
