// RUN: %clang -cc1 -triple %itanium_abi_triple %s -debug-info-kind=limited -fdynamic-debugging -o %t \
// RUN:    -emit-llvm --save-dynamic-debugging-temps --discard-dynamic-debugging-debug-module
// RUN: FileCheck %s --check-prefix=INNER < %t.dyndbg.1.inner.ll
// RUN: FileCheck %s --check-prefix=OUTER < %t.dyndbg.2.outer.ll

// Test global variables get expected linkage and names in the dyndbg inner
// and outer modules. Global data is stored in the outer module and referenced
// from the inner module.
//
// The external global variables are simply referenced by inner.
//
// The internal (static) global variables get external aliases in outer
// and those are referenced from inner.
//
// The internal functions _ZTW1d (thread-local wrapper routine for d) and
// _GLOBAL__sub_I_symbols_globals, __cxx_global_var_init are promoted too (get
// global aliases).

int a = 1;
// OUTER-DAG: @a = global i32 1, align 4
// INNER-DAG: @a = external global i32, align 4

extern int b;
// OUTER-DAG: @b = external global i32, align 4
// INNER-DAG: @b = external global i32, align 4

inline int c = 2;
// OUTER-DAG: $c = comdat any
// OUTER-DAG: @c = linkonce_odr global i32 2, comdat, align 4
// INNER-DAG: @c = external global i32, align 4

thread_local int d = 3;
// OUTER-DAG: @d = thread_local global i32 3, align 4
// INNER-DAG: @d = external thread_local global i32, align 4
//
// OUTER-DAG: $_ZTW1d = comdat any
// OUTER-DAG: define weak_odr hidden noundef ptr @_ZTW1d() #[[#]] comdat
// INNER-DAG: $__dyndbg._ZTW1d = comdat any
// INNER-DAG: declare hidden noundef ptr @_ZTW1d()
// INNER-DAG: define weak_odr hidden noundef ptr @__dyndbg._ZTW1d() #[[#]] comdat

struct S { int a, b; float c, d; } e {0, 100, 4.f, 5.f};
// OUTER-DAG: @e = global %struct.S { i32 0, i32 100, float 4.000000e+00, float 5.000000e+00 }, align 4
// INNER-DAG: @e = external global %struct.S, align 4

inline S f {0, 100, 4.f, 5.f};
// OUTER-DAG: $f = comdat any
// OUTER-DAG: @f = linkonce_odr global %struct.S { i32 0, i32 100, float 4.000000e+00, float 5.000000e+00 }, comdat, align 4
// INNER-DAG: @f = external global %struct.S, align 4

int fun() {
    static int g = 0;
    return g;
}
// OUTER-DAG: @_ZZ3funvE1g = internal global i32 0, align 4
// OUTER-DAG: @_ZZ3funvE1g.dyndbg.[[hash:[0-9A-Z]+]] = hidden alias i32, ptr @_ZZ3funvE1g
// INNER-DAG: @_ZZ3funvE1g.dyndbg.[[hash:[0-9A-Z]+]] = external hidden global i32, align 4

static int h = 1;
// OUTER-DAG: @_ZL1h = internal global i32 1, align 4
// OUTER-DAG: @_ZL1h.dyndbg.[[hash]] = hidden alias i32, ptr @_ZL1h
// INNER-DAG: @_ZL1h.dyndbg.[[hash]] = external hidden global i32, align 4

__attribute__((nodebug)) int use = a + b + c + d + e.a + f.a + h;

// OUTER-DAG: define internal void @__cxx_global_var_init()
// OUTER-DAG: @__cxx_global_var_init.dyndbg.[[hash]] = hidden alias void (), ptr @__cxx_global_var_init
// INNER-DAG: declare hidden void @__cxx_global_var_init.dyndbg.[[hash]]()

// OUTER-DAG: define internal void @_GLOBAL__sub_I_symbols_globals.cpp()
// OUTER-DAG: @_GLOBAL__sub_I_symbols_globals.cpp.dyndbg.[[hash]] = hidden alias void (), ptr @_GLOBAL__sub_I_symbols_globals.cpp
// INNER-DAG: define hidden void @__dyndbg._GLOBAL__sub_I_symbols_globals.cpp.dyndbg.[[hash]]() #[[#]]
