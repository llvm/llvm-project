// RUN: %clang_cc1 %s -triple x86_64-unknown-unknown -emit-llvm -o - | FileCheck %s
// PR6641

extern "C" int printf(const char *, ...);

struct Foo {
 Foo() : iFoo (2) {
  printf("%p\n", this);
 }
 int iFoo;
};


typedef Foo (*T)[3][4];

T bar() {
 return new Foo[2][3][4];
}

T bug(int i) {
  return new Foo[i][3][4];
}

void pr(T a) {
  for (int i = 0; i < 3; i++)
   for (int j = 0; j < 4; j++)
     printf("%p\n", a[i][j]);
}

Foo *test() {
  return new Foo[5];
}

int main() {
 T f =  bar();
 pr(f);
 f = bug(3);
 pr(f);

 Foo * g = test();
 for (int i = 0; i < 5; i++)
 printf("%d\n", g[i].iFoo);
 return 0;
}

// CHECK: call noalias noundef nonnull i8* @_Znam
// CHECK: call noalias noundef nonnull i8* @_Znam
// CHECK: call noalias noundef nonnull i8* @_Znam
