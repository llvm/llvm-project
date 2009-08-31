// RUN: clang-cc -emit-llvm %s -o %t -triple=x86_64-apple-darwin9 &&

struct A {
  int i;
};

// RUN: grep "@a = global i64 -1" %t &&
int A::* a;

// RUN: grep "@aa = global \[2 x i64\] \[i64 -1, i64 -1\], align 8" %t &&
int A::* aa[2];

// RUN: grep "@aaa = global \[2 x \[2 x i64\]\] \[\[2 x i64\] \[i64 -1, i64 -1\], \[2 x i64\] \[i64 -1, i64 -1\]\]" %t &&
int A::* aaa[2][2];

void f() {
  // RUN: grep "%.* = icmp ne i64 %.*, -1" %t
  if (a) { }
  
  // FIXME: This doesn't yet work
//  if (a != 0) { }
    
}

