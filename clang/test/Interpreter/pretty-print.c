// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
// RUN: cat %s | clang-repl -Xcc -xc  | FileCheck %s
// RUN: cat %s | clang-repl -Xcc -std=c++11 | FileCheck %s

// UNSUPPORTED: hwasan

const char* c_str = "Hello, world!"; c_str

char c = 'a'; c
// CHECK: (char) 'a'

c_str = "Goodbye, world!"; c_str
// CHECK-NEXT: (const char *) "Goodbye, world!"

const char* c_null_str = 0; c_null_str
// CHECK-NEXT: (const char *) nullptr

"Hello, world"
// CHECK-NEXT: (const char[13]) "Hello, world"

int x = 42; x
// CHECK-NEXT: (int) 42

&x
// CHECK-NEXT: (int *) @0x{{[0-9a-f]+}}

x - 2
// CHECK-NEXT: (int) 40

float f = 4.2f; f
// CHECK-NEXT: (float) 4.20000f

double d = 4.21; d
// CHECK-NEXT: (double) 4.21000000000

struct S1{} s1; s1
// CHECK-NEXT: (S1 &) @0x{{[0-9a-f]+}}

S1{}
// CHECK-NEXT: (S1) @0x{{[0-9a-f]+}}

struct S2 {int d;} E = {22}; E
// CHECK-NEXT: (struct S2 &) @0x{{[0-9a-f]+}}
E.d
// CHECK-NEXT: (int) 22

// Arrays.

int arr[3] = {1,2,3}; arr
// CHECK-NEXT: (int[3]) { 1, 2, 3 }

int foo() { return 42; } foo()
// CHECK-NEXT: (int) 42

void bar() {} bar()

struct ConstLiteral{};
const char * caas__runtime__PrintValueRuntime(const struct ConstLiteral *) { \
  return "ConstLiteral";                                                \
}
struct ConstLiteral CL; CL
// CHECK-NEXT: ConstLiteral

struct Point{int x; int y};
const char * caas__runtime__PrintValueRuntime(const struct Point *p) { \
  char[11 + 11 + 4 + 1] result;                                        \
  sprintf(result, "(%d, %d)", p->x, p->y);                             \
  return strdup(result);                                               \
}

Point P {1,2}; P
// CHECK-NEXT: (1, 2)

%quit
