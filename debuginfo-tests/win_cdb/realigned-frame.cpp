// RUN: %clang_cl %s -o %t.exe -fuse-ld=lld -Z7
// RUN: grep DE[B]UGGER: %s | sed -e 's/.*DE[B]UGGER: //' > %t.script
// RUN: %cdb -cf %t.script %t.exe | FileCheck %s --check-prefixes=DEBUGGER,CHECK

// From https://llvm.org/pr38857, where we had issues with stack realignment.

struct Foo {
  int x = 42;
  int __declspec(noinline) foo();
  void __declspec(noinline) bar(int *a, int *b, double *c);
};
int Foo::foo() {
  int a = 1;
  int b = 2;
  double __declspec(align(32)) force_alignment = 0.42;
  bar(&a, &b, &force_alignment);
  // DEBUGGER: g
  // DEBUGGER: .frame 1
  // DEBUGGER: dv
  // CHECK: a = 0n1
  // CHECK: b = 0n2
  // CHECK: force_alignment = 0.41999{{.*}}
  // DEBUGGER: q
  x += (int)force_alignment;
  return x;
}
void Foo::bar(int *a, int *b, double *c) {
  __debugbreak();
  *c += *a + *b;
}
int main() {
  Foo o;
  o.foo();
}
