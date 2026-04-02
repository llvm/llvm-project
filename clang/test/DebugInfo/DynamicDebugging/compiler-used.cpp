// RUN: %clang -cc1 -triple %itanium_abi_triple %s -debug-info-kind=limited -fdynamic-debugging -o %t \
// RUN:    -emit-llvm --save-dynamic-debugging-temps --discard-dynamic-debugging-debug-module
// RUN: FileCheck %s --check-prefix=OUTER < %t.dyndbg.2.outer.ll

// Test discardable symbols are added to the @llvm.compiler.used global,
// which prevents them being discarded (including by globalopt replacing) them
// with their external linkage aliases.

// OUTER: @llvm.compiler.used = appending global [[[#]] x ptr]
// OUTER-SAME: [
// OUTER-SAME:  ptr @__cxx_global_var_init,
// OUTER-SAME:  ptr @_ZL12internal_funv,
// OUTER-SAME:  ptr @_Z14internal_fun_2v,
// OUTER-SAME:  ptr @_GLOBAL__sub_I_compiler_used.cpp,
// OUTER-SAME:  ptr @_ZL12used_by_init,
// OUTER-SAME:  ptr @odrweak,
// OUTER-SAME:  ptr @_ZL8internal,
// OUTER-SAME:  ptr @_ZZ14internal_fun_2vE8internal
// OUTER-SAME: ],
// OUTER-SAME: section "llvm.metadata"

// 'external' has external linkage; it's not discardable if unused, so it
// doesn't need to be added to compiler-used.
int external = 1;
// 'odrweak' and 'internal' are both discardable and may be only referenced
// from the inner module, so we must keep them.
inline int odrweak = 2;
static int internal = 1;
// 'unused_internal' has internal linkage and is unused so we don't need to
// preserve it (it isn't referenced by the outer or inner module).
static int internal_fun() { static int unused_internal = 0; return 0; }
inline int internal_fun_2() { static int internal = 0; return internal; }

// Use those globals so they're not omitted by Clang. Don't use [[gnu::used]]
// because that populates the compiler-used global. This global itself is
// unused in user code but it _is_ used in __cxx_global_var_init.
static int used_by_init = external + odrweak + internal + internal_fun() + internal_fun_2();
