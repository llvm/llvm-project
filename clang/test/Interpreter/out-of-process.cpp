// REQUIRES: host-supports-jit, x86_64-linux
// RUN: cat %s | clang-repl | FileCheck %s
// RUN: cat %s | clang-repl -oop-executor | FileCheck %s

extern "C" int printf(const char *, ...);

// Test code undo.
int x1 = 0;
int x2 = 42;
%undo
int x2 = 24;
auto r1 = printf("x1 = %d\n", x1);
// CHECK: x1 = 0
auto r2 = printf("x2 = %d\n", x2);
// CHECK-NEXT: x2 = 24
int foo() { return 1; }
%undo
int foo() { return 2; }
auto r3 = printf("foo() = %d\n", foo());
// CHECK-NEXT: foo() = 2
inline int bar() { return 42;}
auto r4 = bar();
%undo
auto r5 = bar();

// Test weak execution.
int __attribute__((weak)) weak_bar() { return 42; }
auto r6 = printf("bar() = %d\n", weak_bar());
// CHECK: bar() = 42
int a = 12;
static __typeof(a) b __attribute__((__weakref__("a")));
int c = b;

// Test dynamic libraries.
extern "C" int ultimate_answer;
extern "C" int calculate_answer();
%lib libdynamic-library-test.so
printf("Return value: %d\n", calculate_answer());
// CHECK: Return value: 5
printf("Variable: %d\n", ultimate_answer);
// CHECK-NEXT: Variable: 42

// Test lambdas.
auto l1 = []() { printf("ONE\n"); return 42; };
auto l2 = []() { printf("TWO\n"); return 17; };
auto lambda_r1 = l1();
// CHECK: ONE
auto lambda_r2 = l2();
// CHECK: TWO
auto lambda_r3 = l2();
// CHECK: TWO

// Test multiline.
int i = \
  12;
printf("i=%d\n", i);
// CHECK: i=12
void f(int x) \ 
{                                               \
  printf("x=\
          %d", x); \
}
f(i);
// CHECK: x=12

// Test global destructor.
struct D { float f = 1.0; D *m = nullptr; D(){} ~D() { printf("D[f=%f, m=0x%llx]\n", f, reinterpret_cast<unsigned long long>(m)); }} d;
// CHECK: D[f=1.000000, m=0x0]

%quit
