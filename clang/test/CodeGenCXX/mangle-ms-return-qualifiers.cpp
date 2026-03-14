// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s

void a1() {}
// CHECK: "?a1@@YAXXZ"

int a2() { return 0; }
// CHECK: "?a2@@YAHXZ"

const int a3() { return 0; }
// CHECK: "?a3@@YA?BHXZ"

volatile int a4() { return 0; }
// CHECK: "?a4@@YA?CHXZ"

const volatile int a5() { return 0; }
// CHECK: "?a5@@YA?DHXZ"

float a6() { return 0.0f; }
// CHECK: "?a6@@YAMXZ"

int *b1() { return 0; }
// CHECK: "?b1@@YAPAHXZ"

const char *b2() { return 0; }
// CHECK: "?b2@@YAPBDXZ"

float *b3() { return 0; }
// CHECK: "?b3@@YAPAMXZ"

const float *b4() { return 0; }
// CHECK: "?b4@@YAPBMXZ"

volatile float *b5() { return 0; }
// CHECK: "?b5@@YAPCMXZ"

const volatile float *b6() { return 0; }
// CHECK: "?b6@@YAPDMXZ"

float &b7() { return *(float*)0; }
// CHECK: "?b7@@YAAAMXZ"

const float &b8() { return *(float*)0; }
// CHECK: "?b8@@YAABMXZ"

volatile float &b9() { return *(float*)0; }
// CHECK: "?b9@@YAACMXZ"

const volatile float &b10() { return *(float*)0; }
// CHECK: "?b10@@YAADMXZ"

const char** b11() { return 0; }
// CHECK: "?b11@@YAPAPBDXZ"

class A {};

A c1() { return A(); }
// CHECK: "?c1@@YA?AVA@@XZ"

const A c2() { return A(); }
// CHECK: "?c2@@YA?BVA@@XZ"

volatile A c3() { return A(); }
// CHECK: "?c3@@YA?CVA@@XZ"

const volatile A c4() { return A(); }
// CHECK: "?c4@@YA?DVA@@XZ"

const A* c5() { return 0; }
// CHECK: "?c5@@YAPBVA@@XZ"

volatile A* c6() { return 0; }
// CHECK: "?c6@@YAPCVA@@XZ"

const volatile A* c7() { return 0; }
// CHECK: "?c7@@YAPDVA@@XZ"

A &c8() { return *(A*)0; }
// CHECK: "?c8@@YAAAVA@@XZ"

const A &c9() { return *(A*)0; }
// CHECK: "?c9@@YAABVA@@XZ"

volatile A &c10() { return *(A*)0; }
// CHECK: "?c10@@YAACVA@@XZ"

const volatile A &c11() { return *(A*)0; }
// CHECK: "?c11@@YAADVA@@XZ"

template<typename T> class B {};

B<int> d1() { return B<int>(); }
// CHECK: "?d1@@YA?AV?$B@H@@XZ"

B<const char*> d2() {return B<const char*>(); }
// CHECK: "?d2@@YA?AV?$B@PBD@@XZ"

B<A> d3() {return B<A>(); }
// CHECK: "?d3@@YA?AV?$B@VA@@@@XZ"

B<A>* d4() { return 0; }
// CHECK: "?d4@@YAPAV?$B@VA@@@@XZ"

const B<A>* d5() { return 0; }
// CHECK: "?d5@@YAPBV?$B@VA@@@@XZ"

volatile B<A>* d6() { return 0; }
// CHECK: "?d6@@YAPCV?$B@VA@@@@XZ"

const volatile B<A>* d7() { return 0; }
// CHECK: "?d7@@YAPDV?$B@VA@@@@XZ"

B<A>& d8() { return *(B<A>*)0; }
// CHECK: "?d8@@YAAAV?$B@VA@@@@XZ"

const B<A>& d9() { return *(B<A>*)0; }
// CHECK: "?d9@@YAABV?$B@VA@@@@XZ"

volatile B<A>& d10() { return *(B<A>*)0; }
// CHECK: "?d10@@YAACV?$B@VA@@@@XZ"

const volatile B<A>& d11() { return *(B<A>*)0; }
// CHECK: "?d11@@YAADV?$B@VA@@@@XZ"

enum Enum { DEFAULT };

Enum e1() { return DEFAULT; }
// CHECK: "?e1@@YA?AW4Enum@@XZ"

const Enum e2() { return DEFAULT; }
// CHECK: "?e2@@YA?BW4Enum@@XZ"

Enum* e3() { return 0; }
// CHECK: "?e3@@YAPAW4Enum@@XZ"

Enum& e4() { return *(Enum*)0; }
// CHECK: "?e4@@YAAAW4Enum@@XZ"

struct S {};

struct S f1() { struct S s; return s; }
// CHECK: "?f1@@YA?AUS@@XZ"

const struct S f2() { struct S s; return s; }
// CHECK: "?f2@@YA?BUS@@XZ"

struct S* f3() { return 0; }
// CHECK: "?f3@@YAPAUS@@XZ"

const struct S* f4() { return 0; }
// CHECK: "?f4@@YAPBUS@@XZ"

const volatile struct S* f5() { return 0; }
// CHECK: "?f5@@YAPDUS@@XZ"

struct S& f6() { return *(struct S*)0; }
// CHECK: "?f6@@YAAAUS@@XZ"

struct S* const f7() { return 0; }
// CHECK: "?f7@@YAQAUS@@XZ"

int S::* f8() { return 0; }
// CHECK: "?f8@@YAPQS@@HXZ"

int S::* const f9() { return 0; }
// CHECK: "?f9@@YAQQS@@HXZ"

int S::* __restrict f10() { return 0; }
// CHECK: "?f10@@YAPIQS@@HXZ"

int S::* const __restrict f11() { return 0; }
// CHECK: "?f11@@YAQIQS@@HXZ"

typedef int (*function_pointer)(int);

function_pointer g1() { return 0; }
// CHECK: "?g1@@YAP6AHH@ZXZ"

const function_pointer g2() { return 0; }
// CHECK: "?g2@@YAQ6AHH@ZXZ"

function_pointer* g3() { return 0; }
// CHECK: "?g3@@YAPAP6AHH@ZXZ"

const function_pointer* g4() { return 0; }
// CHECK: "?g4@@YAPBQ6AHH@ZXZ"

extern int &z;
int & __restrict h1() { return z; }
// CHECK: "?h1@@YAAIAHXZ"
