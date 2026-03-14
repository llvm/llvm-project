// RUN: %clang_cc1 %s -triple=powerpc-ibm-aix-xcoff -mtocdata=h,g,f,e,d,c,b,a,globalOneWithAlias,globalTwoWithAlias,ll,t3 -verify -emit-llvm -o - | FileCheck %s -check-prefix=CHECK --match-full-lines
// RUN: %clang_cc1 %s -triple=powerpc-ibm-aix-xcoff -mtocdata -verify=none -emit-llvm -o - | FileCheck %s -check-prefix=CHECK --match-full-lines

// none-no-diagnostics

struct large_struct {
  int x;
  short y;
  short z;
  char c;
};

struct large_struct a;                      // expected-warning {{-mtocdata option is ignored for a because variable is larger than a pointer}}
long long b = 5;                            // expected-warning {{-mtocdata option is ignored for b because variable is larger than a pointer}}
int __attribute__((aligned(128))) c = 0;    // expected-warning {{-mtocdata option is ignored for c because variable is aligned wider than a pointer}}
double d = 2.5;                             // expected-warning {{-mtocdata option is ignored for d because variable is larger than a pointer}}
int e __attribute__((section("foo"))) = 10; // expected-warning {{-mtocdata option is ignored for e because variable has a section attribute}}
__thread int f;                             // expected-warning {{-mtocdata option is ignored for f because of thread local storage}}

struct SomeStruct;
extern struct SomeStruct g;                 // expected-warning {{-mtocdata option is ignored for g because of incomplete type}}

extern int h[];                             // expected-warning {{-mtocdata option is ignored for h because of incomplete type}}

struct ty3 {
  int A;
  char C[];
};
struct ty3 t3 = { 4, "fo" }; // expected-warning {{-mtocdata option is ignored for t3 because it contains a flexible array member}}

int globalOneWithAlias = 10;
__attribute__((__alias__("globalOneWithAlias"))) extern int aliasOne; // expected-warning {{-mtocdata option is ignored for globalOneWithAlias because the variable has an alias}}
__attribute__((__alias__("globalTwoWithAlias"))) extern int aliasTwo; // expected-warning {{-mtocdata option is ignored for globalTwoWithAlias because the variable has an alias}}
int globalTwoWithAlias = 20;


int func() {
  return a.x+b+c+d+e+f+h[0];
}

struct SomeStruct* getAddress(void) {
  return &g;
}

int test() {
  return globalOneWithAlias + globalTwoWithAlias + aliasOne + aliasTwo;
}

long long test2() {
  static long long ll = 5;
  ll++;
  return ll;
}

// CHECK: @b = global i64 5, align 8
// CHECK: @c = global i32 0, align 128
// CHECK: @d = global double 2.500000e+00, align 8
// CHECK: @e = global i32 10, section "foo", align 4
// CHECK: @globalOneWithAlias = global i32 10, align 4
// CHECK: @globalTwoWithAlias = global i32 20, align 4
// CHECK: @a = global %struct.large_struct zeroinitializer, align 4
// CHECK: @f = thread_local global i32 0, align 4
// CHECK: @h = external global [0 x i32], align 4
// CHECK: @g = external global %struct.SomeStruct, align 1 
// CHECK: @test2.ll = internal global i64 5, align 8
// CHECK: @aliasOne = alias i32, ptr @globalOneWithAlias
// CHECK: @aliasTwo = alias i32, ptr @globalTwoWithAlias
// CHECK-NOT: attributes #0 = { "toc-data" }
