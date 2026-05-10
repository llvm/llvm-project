// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.15.0 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm %s -o - | FileCheck %s
//
// ELF with -fPIC (shared library): default-visibility symbols may be
// COPY-relocated, so alignment must NOT be bumped for them.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -pic-level 2 -emit-llvm %s -o - | FileCheck %s --check-prefix=ELF-PIC
//
// ELF with -fPIE (executable): definitions are dso_local, alignment CAN be bumped.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -pic-level 2 -pic-is-pie -emit-llvm %s -o - | FileCheck %s --check-prefix=ELF-PIE
//
// C++17 inline variables (weak/linkonce_odr linkage):
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=CXX

// Test that getLargeGlobalPreferredAlign does not bump alignment for variables
// where the compiler cannot safely control placement. This mirrors the
// conditions in GlobalObject::canIncreaseAlignment.

// A 128-byte struct of doubles (natural alignment 8)
struct S128Doubles {
    double m11,m12,m13,m14;
    double m21,m22,m23,m24;
    double m31,m32,m33,m34;
    double m41,m42,m43,m44;
};

// A 128-byte struct of chars (natural alignment 1)
struct S128 {
    char Buffer[128];
};

// A 64-byte struct of chars (natural alignment 1)
struct S64 {
    char Buffer[64];
};

// A 16-byte struct of chars (natural alignment 1)
struct S16 {
    char Buffer[16];
};

// --- Extern declarations: no definition, alignment must NOT be bumped ---
extern struct S128Doubles extern_s128doubles;
extern struct S128 extern_s128;
extern struct S64 extern_s64;
extern struct S16 extern_s16;

// CHECK-DAG: @extern_s128doubles = external {{(dso_local )?}}global %struct.S128Doubles, align 8
// CHECK-DAG: @extern_s128 = external {{(dso_local )?}}global %struct.S128, align 1
// CHECK-DAG: @extern_s64 = external {{(dso_local )?}}global %struct.S64, align 1
// CHECK-DAG: @extern_s16 = external {{(dso_local )?}}global %struct.S16, align 1

// --- Strong definitions: alignment CAN be bumped ---
struct S128Doubles defined_s128doubles = {0};
struct S128 defined_s128 = {0};
struct S64 defined_s64 = {0};
struct S16 defined_s16 = {0};

// CHECK-DAG: @defined_s128doubles = {{(dso_local )?}}global %struct.S128Doubles zeroinitializer, align 16
// CHECK-DAG: @defined_s128 = {{(dso_local )?}}global %struct.S128 zeroinitializer, align 16
// CHECK-DAG: @defined_s64 = {{(dso_local )?}}global %struct.S64 zeroinitializer, align 8
// CHECK-DAG: @defined_s16 = {{(dso_local )?}}global %struct.S16 zeroinitializer, align 8

// --- Weak definitions: alignment must NOT be bumped ---
__attribute__((weak)) struct S128 weak_s128 = {0};
__attribute__((weak)) struct S64 weak_s64 = {0};

// CHECK-DAG: @weak_s128 = weak {{(dso_local )?}}global %struct.S128 zeroinitializer, align 1
// CHECK-DAG: @weak_s64 = weak {{(dso_local )?}}global %struct.S64 zeroinitializer, align 1

#ifdef __APPLE__
__attribute__((section("__DATA,.mysect"))) struct S128 section_s128 = {0};
__attribute__((section("__DATA,.mysect"))) struct S64 section_s64 = {0};
#else
__attribute__((section(".mysect"))) struct S128 section_s128 = {0};
__attribute__((section(".mysect"))) struct S64 section_s64 = {0};
#endif

// CHECK-DAG: @section_s128 = {{(dso_local )?}}global %struct.S128 zeroinitializer, section "{{[^"]+}}", align 1
// CHECK-DAG: @section_s64 = {{(dso_local )?}}global %struct.S64 zeroinitializer, section "{{[^"]+}}", align 1

// --- Tentative definitions: alignment CAN be bumped (these are strong defs) ---
// Tentative definitions are a C-only concept (not valid in C++).
#ifndef __cplusplus
struct S128 tentative_s128;
struct S64 tentative_s64;

// CHECK-DAG: @tentative_s128 = {{(dso_local )?}}global %struct.S128 zeroinitializer, align 16
// CHECK-DAG: @tentative_s64 = {{(dso_local )?}}global %struct.S64 zeroinitializer, align 8
#endif

// --- ELF with -fPIC: default-visibility defs must NOT be bumped (COPY relocation risk) ---
// ELF-PIC-DAG: @defined_s128 = global %struct.S128 zeroinitializer, align 1
// ELF-PIC-DAG: @defined_s64 = global %struct.S64 zeroinitializer, align 1
// ELF-PIC-DAG: @tentative_s128 = global %struct.S128 zeroinitializer, align 1
// ELF-PIC-DAG: @tentative_s64 = global %struct.S64 zeroinitializer, align 1
// Hidden visibility is dso_local even with -fPIC, so alignment CAN be bumped.
// ELF-PIC-DAG: @hidden_s128 = hidden global %struct.S128 zeroinitializer, align 16
// ELF-PIC-DAG: @hidden_s64 = hidden global %struct.S64 zeroinitializer, align 8

// --- ELF with -fPIE: definitions are dso_local, alignment CAN be bumped ---
// ELF-PIE-DAG: @defined_s128 = dso_local global %struct.S128 zeroinitializer, align 16
// ELF-PIE-DAG: @defined_s64 = dso_local global %struct.S64 zeroinitializer, align 8
// ELF-PIE-DAG: @tentative_s128 = dso_local global %struct.S128 zeroinitializer, align 16
// ELF-PIE-DAG: @tentative_s64 = dso_local global %struct.S64 zeroinitializer, align 8

// Hidden visibility: safe to bump on all configurations.
__attribute__((visibility("hidden"))) struct S128 hidden_s128 = {0};
__attribute__((visibility("hidden"))) struct S64 hidden_s64 = {0};

// CHECK-DAG: @hidden_s128 = {{(dso_local )?}}hidden global %struct.S128 zeroinitializer, align 16
// CHECK-DAG: @hidden_s64 = {{(dso_local )?}}hidden global %struct.S64 zeroinitializer, align 8

void use(void *);

#ifdef __cplusplus
// --- C++17 inline variables: weak (linkonce_odr) linkage, must NOT be bumped ---
inline S128 inline_s128 = {};
inline S64 inline_s64 = {};

// CXX-DAG: @inline_s128 = linkonce_odr {{(dso_local )?}}global %struct.S128 zeroinitializer, comdat, align 1
// CXX-DAG: @inline_s64 = linkonce_odr {{(dso_local )?}}global %struct.S64 zeroinitializer, comdat, align 1

// constexpr static data members (implicitly inline in C++17) also weak
struct Holder {
    static constexpr S128 member = {};
};

// CXX-DAG: @_ZN6Holder6memberE = linkonce_odr {{(dso_local )?}}constant %struct.S128 zeroinitializer, comdat, align 1

// Strong definitions in C++ should still be bumped.
// CXX-DAG: @defined_s128 = {{(dso_local )?}}global %struct.S128 zeroinitializer, align 16
// CXX-DAG: @defined_s64 = {{(dso_local )?}}global %struct.S64 zeroinitializer, align 8

// --- Static locals in inline functions: linkonce_odr, must NOT be bumped ---
// (Itanium ABI 5.2.2: COMDAT group, linker picks any copy)
inline void inline_func() {
    static S128 local_s128 = {};
    use(&local_s128);
}

// CXX-DAG: @_ZZ11inline_funcvE10local_s128 = linkonce_odr {{(dso_local )?}}global %struct.S128 zeroinitializer, comdat, align 1

// Static locals in non-inline functions: internal linkage, CAN be bumped.
void regular_func() {
    static S128 local_s128 = {};
    use(&local_s128);
}

// CXX-DAG: @_ZZ12regular_funcvE10local_s128 = internal global %struct.S128 zeroinitializer, align 16

// --- Template variables: implicit instantiation gets linkonce_odr, must NOT be bumped ---
template<typename T> T tmpl_var = {};
template S128 tmpl_var<S128>; // explicit instantiation definition

// CXX-DAG: @_Z8tmpl_varI4S128E = weak_odr {{(dso_local )?}}global %struct.S128 zeroinitializer, comdat, align 1

// Implicit instantiation
// CXX-DAG: @_Z8tmpl_varI3S64E = linkonce_odr {{(dso_local )?}}global %struct.S64 zeroinitializer, comdat, align 1
#endif

void test(void) {
    use(&extern_s128doubles);
    use(&extern_s128);
    use(&extern_s64);
    use(&extern_s16);
    use(&defined_s128doubles);
    use(&defined_s128);
    use(&defined_s64);
    use(&defined_s16);
    use(&weak_s128);
    use(&weak_s64);
    use(&section_s128);
    use(&section_s64);
#ifndef __cplusplus
    use(&tentative_s128);
    use(&tentative_s64);
#endif
    use(&hidden_s128);
    use(&hidden_s64);
#ifdef __cplusplus
    use(&inline_s128);
    use(&inline_s64);
    use((void *)&Holder::member);
    inline_func();
    regular_func();
    use(&tmpl_var<S128>);
    use(&tmpl_var<S64>);
#endif
}
