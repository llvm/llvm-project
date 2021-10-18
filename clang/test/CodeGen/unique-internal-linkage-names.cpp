// This test checks if internal linkage symbols get unique names with
// -funique-internal-linkage-names option.
// RUN: %clang_cc1 -triple x86_64 -x c++ -S -emit-llvm -o - < %s | FileCheck %s --check-prefix=PLAIN
// RUN: %clang_cc1 -triple x86_64 -x c++  -S -emit-llvm -funique-internal-linkage-names -o - < %s | FileCheck %s --check-prefix=UNIQUE

static int glob;
static int foo() {
  return 0;
}

int (*bar())() {
  return foo;
}

int getGlob() {
  return glob;
}

// Function local static variable and anonymous namespace namespace variable.
namespace {
int anon_m;
int getM() {
  return anon_m;
}
} // namespace

int retAnonM() {
  static int fGlob;
  return getM() + fGlob;
}

// Multiversioning symbols
__attribute__((target("default"))) static int mver() {
  return 0;
}

__attribute__((target("sse4.2"))) static int mver() {
  return 1;
}

int mver_call() {
  return mver();
}

namespace {
class A {
public:
  A() {}
  ~A() {}
};
}

void test() {
  A a;
}

// PLAIN: @_ZL4glob = internal global
// PLAIN: @_ZZ8retAnonMvE5fGlob = internal global
// PLAIN: @_ZN12_GLOBAL__N_16anon_mE = internal global
// PLAIN: define internal i32 @_ZL3foov()
// PLAIN: define internal i32 @_ZN12_GLOBAL__N_14getMEv
// PLAIN: define internal i32 ()* @_ZL4mverv.resolver()
// PLAIN: define internal void @_ZN12_GLOBAL__N_11AC1Ev
// PLAIN: define internal void @_ZN12_GLOBAL__N_11AD1Ev
// PLAIN: define internal i32 @_ZL4mverv()
// PLAIN: define internal i32 @_ZL4mverv.sse4.2()
// PLAIN-NOT: "sample-profile-suffix-elision-policy"
// UNIQUE: @_ZL4glob = internal global
// UNIQUE: @_ZZ8retAnonMvE5fGlob = internal global
// UNIQUE: @_ZN12_GLOBAL__N_16anon_mE = internal global
// UNIQUE: define internal i32 @_ZL3foov.[[MODHASH:__uniq.[0-9]+]]() #[[#ATTR:]] {
// UNIQUE: define internal i32 @_ZN12_GLOBAL__N_14getMEv.[[MODHASH]]
// UNIQUE: define internal i32 ()* @_ZL4mverv.[[MODHASH]].resolver()
// UNIQUE: define internal void @_ZN12_GLOBAL__N_11AC1Ev.__uniq.68358509610070717889884130747296293671
// UNIQUE: define internal void @_ZN12_GLOBAL__N_11AD1Ev.__uniq.68358509610070717889884130747296293671
// UNIQUE: define internal i32 @_ZL4mverv.[[MODHASH]]()
// UNIQUE: define internal i32 @_ZL4mverv.[[MODHASH]].sse4.2
// UNIQUE: attributes #[[#ATTR]] = { {{.*}}"sample-profile-suffix-elision-policy"{{.*}} }
