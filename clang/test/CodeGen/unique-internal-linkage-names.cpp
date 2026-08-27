// This test checks if internal linkage symbols get unique names with
// -funique-internal-linkage-names option.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -x c++ -emit-llvm -o - < %s | FileCheck %s --check-prefix=PLAIN
// Note: gets a module path as '-' instead of a full path to the source test file that allows using the same module path for any build environment. 
// RUN: %clang_cc1 -triple x86_64-linux-gnu -x c++  -emit-llvm -funique-internal-linkage-names -o - < %s | FileCheck %s --check-prefix=UNIQUE
// Check with path mapping. The unique id must be reproducable for the specified target on any build system.
// Note: we expect path normalization for the specified prefix map in favor of the target system accordingly.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -x c++  -emit-llvm -funique-internal-linkage-names -fmacro-prefix-map=%S/=/repro/./src/path/ %s -o - | FileCheck %s --check-prefix=UNIQUE-PATH-MAP-LINUX
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -x c++  -emit-llvm -funique-internal-linkage-names -fmacro-prefix-map=%S/=/repro/src/./path/ %s -o - | FileCheck %s --check-prefix=UNIQUE-PATH-MAP-WINDOWS

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

// Check a static function with an asm label must keep original name.
static int asm_label() asm("custom_label");
static int asm_label() { return 0; }
int call_asm_label() { return asm_label(); }

// PLAIN: @_ZL4glob = internal global
// PLAIN: @_ZZ8retAnonMvE5fGlob = internal global
// PLAIN: @_ZN12_GLOBAL__N_16anon_mE = internal global
// PLAIN: define internal noundef i32 @_ZL3foov()
// PLAIN: define internal noundef i32 @_ZN12_GLOBAL__N_14getMEv
// PLAIN: define internal ptr @_ZL4mverv.resolver()
// PLAIN: define internal void @_ZN12_GLOBAL__N_11AC1Ev
// PLAIN: define internal void @_ZN12_GLOBAL__N_11AD1Ev
// PLAIN: define internal noundef i32 @custom_label()
// PLAIN: define internal noundef i32 @_ZL4mverv()
// PLAIN: define internal noundef i32 @_ZL4mverv.sse4.2()
// PLAIN-NOT: "sample-profile-suffix-elision-policy"
// UNIQUE: @_ZL4glob = internal global
// UNIQUE: @_ZZ8retAnonMvE5fGlob = internal global
// UNIQUE: @_ZN12_GLOBAL__N_16anon_mE = internal global
// UNIQUE: define internal noundef i32 @_ZL3foov.[[MODHASH:__uniq.[0-9]+]]() #[[#ATTR:]] {
// UNIQUE: define internal noundef i32 @_ZN12_GLOBAL__N_14getMEv.[[MODHASH]]
// UNIQUE: define internal ptr @_ZL4mverv.[[MODHASH]].resolver()
// UNIQUE: define internal void @_ZN12_GLOBAL__N_11AC1Ev.__uniq.68358509610070717889884130747296293671
// UNIQUE: define internal void @_ZN12_GLOBAL__N_11AD1Ev.__uniq.68358509610070717889884130747296293671
// UNIQUE: define internal noundef i32 @custom_label()
// UNIQUE: define internal noundef i32 @_ZL4mverv.[[MODHASH]]()
// UNIQUE: define internal noundef i32 @_ZL4mverv.[[MODHASH]].sse4.2
// UNIQUE: attributes #[[#ATTR]] = { {{.*}}"sample-profile-suffix-elision-policy"{{.*}} }

// Expected module path and unique ID
// /repro/src/path/unique-internal-linkage-names.cpp => __uniq.5283619504002921413211664429594652319

// UNIQUE-PATH-MAP-LINUX: @_ZL4glob = internal global
// UNIQUE-PATH-MAP-LINUX: @_ZZ8retAnonMvE5fGlob = internal global
// UNIQUE-PATH-MAP-LINUX: @_ZN12_GLOBAL__N_16anon_mE = internal global
// UNIQUE-PATH-MAP-LINUX: define internal noundef i32 @_ZL3foov.__uniq.5283619504002921413211664429594652319() #[[#ATTR:]] {
// UNIQUE-PATH-MAP-LINUX: define internal noundef i32 @_ZN12_GLOBAL__N_14getMEv.__uniq.5283619504002921413211664429594652319
// UNIQUE-PATH-MAP-LINUX: define internal ptr @_ZL4mverv.__uniq.5283619504002921413211664429594652319.resolver()
// UNIQUE-PATH-MAP-LINUX: define internal void @_ZN12_GLOBAL__N_11AC1Ev.__uniq.5283619504002921413211664429594652319
// UNIQUE-PATH-MAP-LINUX: define internal void @_ZN12_GLOBAL__N_11AD1Ev.__uniq.5283619504002921413211664429594652319
// UNIQUE-PATH-MAP-LINUX: define internal noundef i32 @_ZL4mverv.__uniq.5283619504002921413211664429594652319()
// UNIQUE-PATH-MAP-LINUX: define internal noundef i32 @_ZL4mverv.__uniq.5283619504002921413211664429594652319.sse4.2
// UNIQUE-PATH-MAP-LINUX: attributes #[[#ATTR]] = { {{.*}}"sample-profile-suffix-elision-policy"{{.*}} }

// Expected module path and unique ID
// \repro\src\path\unique-internal-linkage-names.cpp => __uniq.68451533753012730514350177221027644473

// UNIQUE-PATH-MAP-WINDOWS: @glob = internal global
// UNIQUE-PATH-MAP-WINDOWS: @"?fGlob@?1??retAnonM@@YAHXZ@4HA" = internal global
// UNIQUE-PATH-MAP-WINDOWS: @"?anon_m@?{{.*}}@@3HA" = internal global
// UNIQUE-PATH-MAP-WINDOWS: ret ptr @"?foo@@YAHXZ.__uniq.68451533753012730514350177221027644473"
// UNIQUE-PATH-MAP-WINDOWS: define internal noundef i32 @"?foo@@YAHXZ.__uniq.68451533753012730514350177221027644473"
// UNIQUE-PATH-MAP-WINDOWS: define internal i32 @"?mver@@YAHXZ.__uniq.68451533753012730514350177221027644473.resolver"()
// UNIQUE-PATH-MAP-WINDOWS: define internal noundef ptr @"??0A@?{{.*}}@XZ.__uniq.68451533753012730514350177221027644473"
// UNIQUE-PATH-MAP-WINDOWS: define internal noundef i32 @"?mver@@YAHXZ.__uniq.68451533753012730514350177221027644473"()
// UNIQUE-PATH-MAP-WINDOWS: define internal noundef i32 @"?mver@@YAHXZ.__uniq.68451533753012730514350177221027644473.sse4.2"()
