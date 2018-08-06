// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux -emit-llvm -fsanitize=alignment -fblocks %s -o %t.ll
// RUN: FileCheck -check-prefix=ZEROINIT < %t.ll %s
// RUN: FileCheck -check-prefix=SRCLOC < %t.ll %s
// ZEROINIT-NOT: @{{.+}} = private unnamed_addr global {{.+}} zeroinitializer

struct A {
  A(int);
  int k;
};

struct B : A {
  B();
  B(const B &);
// SRCLOC-DAG: @{{.+}} = private unnamed_addr global {{.+}} @.src, i32 [[@LINE+1]], i32 12 }
  using A::A;
  void f() const;
};

// SRCLOC-DAG: @{{.+}} = private unnamed_addr global {{.+}} @.src, i32 [[@LINE+1]], i32 10 }
B::B() : A(1) {}

void foo() {
  B b(2);
// SRCLOC-DAG: @{{.+}} = private unnamed_addr global {{.+}} @.src, i32 [[@LINE+1]], i32 5 }
  ^{b.f();}();
}
