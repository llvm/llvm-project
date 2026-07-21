// C/C++ behavior of -mloadtime-comment-vars= :
//   codegen.cpp - mangled-name matching and what gets preserved
//   storage.cpp - storage-duration and scope diagnostics
//   diag.c      - volatile / non-string-literal diagnostics (C)
//   init.cpp    - constant-initialization / string-literal diagnostics (C++)
//   list.c      - list parsing: whitespace after a comma, repeated names

// RUN: rm -rf %t && split-file %s %t
//
// RUN: %clang_cc1 -std=c++17 -O2 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=x,_ZN1N1xE,_ZN1N1qE,_ZN1NL3ptrE,_ZN1A1xE,_ZN1B3verE,_ZN1C4infoE \
// RUN:   -emit-llvm -disable-llvm-passes -o - %t/codegen.cpp | FileCheck %t/codegen.cpp
//
// RUN: %clang_cc1 -std=c++17 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=keep,_ZN1N2tlE,_ZL3stl,_ZN1A2tmE,_ZZ1fvE2fn \
// RUN:   -emit-llvm -verify -o - %t/storage.cpp | FileCheck %t/storage.cpp
//
// RUN: %clang_cc1 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=vol_ptr,vol_char,vol_arr,tls_ptr,ind_ptr,const_arr \
// RUN:   -emit-llvm -verify -o - %t/diag.c | FileCheck %t/diag.c
//
// RUN: %clang_cc1 -std=c++17 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=p_ok,arr_ok,p_dyn,p_ind \
// RUN:   -emit-llvm -verify -o - %t/init.cpp | FileCheck %t/init.cpp
//
// The diagnostics are produced by Sema, so they are emitted even when no code
// is generated.
// RUN: %clang_cc1 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=vol_ptr,vol_char,vol_arr,tls_ptr,ind_ptr,const_arr \
// RUN:   -fsyntax-only -verify %t/diag.c
// RUN: %clang_cc1 -std=c++17 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=p_ok,arr_ok,p_dyn,p_ind \
// RUN:   -fsyntax-only -verify %t/init.cpp
//
// RUN: %clang_cc1 -triple powerpc64-ibm-aix \
// RUN:   "-mloadtime-comment-vars=foo, bar" \
// RUN:   -emit-llvm -o - %t/list.c | FileCheck %t/list.c --check-prefix=SPACE
//
// RUN: %clang_cc1 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=foo,foo \
// RUN:   -emit-llvm -o - %t/list.c | FileCheck %t/list.c --check-prefix=DUP

//--- codegen.cpp
// Names are matched against mangled IR symbol names.
// C++ variables use Itanium ABI mangling; C/file-scope statics keep their
// source name.
//
// Mangled names used here:
//   x            -> x             (file-scope, no mangling)
//   N::x         -> _ZN1N1xE
//   N::q         -> _ZN1N1qE
//   N::ptr       -> _ZN1NL3ptrE   (static, internal linkage)
//   A::x         -> _ZN1A1xE
//   B::ver       -> _ZN1B3verE
//   C::info      -> _ZN1C4infoE   (declared only, no definition — skipped)

// 1. File-scope array "x" — no mangling in C++, IR name == source name.
char x[] = "@(#) global x";

namespace N {
char x[] = "@(#) ns x";

// 2. Namespace member "N::x" — mangled as _ZN1N1xE.
char q[] = "@(#) ns q";

// 3. Namespace-scope pointer initialized with a string literal.
//    _ZN1NL3ptrE (N::ptr) is internal (it is a const variable at namespace
//    scope). MustBeEmitted forces it through EmitGlobalVarDefinition.
static const char *ptr = "@(#) ns ptr";
} // namespace N

// 4. Class static member "A::x" — mangled as _ZN1A1xE.
struct A {
  static const char *x;
};
const char *A::x = "@(#) class x";

// 5. Class static member pointer initialized with a string literal.
//    _ZN1B3verE (B::ver).
struct B {
  static const char *ver;
};
const char *B::ver = "@(#) class ver";

// 6. _ZN1C4infoE is in the list but C::info has no definition in this TU —
//    must be silently skipped.
struct C { static const char *info; };

// 7. Invalid type — int must not be tagged regardless of its IR name.
int not_string = 7;

void f() {}

// File-scope x and namespace N::x both matched.
// CHECK-DAG: @x = global [14 x i8] c"@(#) global x\00", align {{[0-9]+}}, !loadtime_comment ![[MD:[0-9]+]]
// CHECK-DAG: @_ZN1N1xE = global [10 x i8] c"@(#) ns x\00", align {{[0-9]+}}, !loadtime_comment ![[MD]]

// N::q matched by mangled name _ZN1N1qE.
// CHECK-DAG: @_ZN1N1qE = global [10 x i8] c"@(#) ns q\00", align {{[0-9]+}}, !loadtime_comment ![[MD]]

// A::x matched by mangled name _ZN1A1xE.
// CHECK-DAG: @[[AX:_ZN1A1xE]] = {{.*}}global ptr @[[AXSTR:.*]], align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @[[AXSTR]] = private unnamed_addr constant [13 x i8] c"@(#) class x\00", align {{[0-9]+}}

// N::ptr (_ZN1NL3ptrE) points to a string literal.
// CHECK-DAG: @_ZN1NL3ptrE = internal global ptr @[[NPTR_STR:.*]], align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @[[NPTR_STR]] = private unnamed_addr constant [{{[0-9]+}} x i8] c"@(#) ns ptr\00", align {{[0-9]+}}

// B::ver (_ZN1B3verE) points to a string literal.
// CHECK-DAG: @_ZN1B3verE = global ptr @[[BVER_STR:.*]], align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @[[BVER_STR]] = private unnamed_addr constant [{{[0-9]+}} x i8] c"@(#) class ver\00", align {{[0-9]+}}

// Invalid type must not be tagged.
// CHECK-NOT: @not_string{{.*}}!loadtime_comment

// C::info has no definition — must not appear.
// CHECK-NOT: @_ZN1C4infoE

// All six selected globals are preserved in llvm.compiler.used.
// CHECK: @llvm.compiler.used = appending global [6 x ptr]
// CHECK-SAME: @x
// CHECK-SAME: @_ZN1N1xE
// CHECK-SAME: @_ZN1N1qE
// CHECK-SAME: @_ZN1NL3ptrE
// CHECK-SAME: @[[AX]]
// CHECK-SAME: @_ZN1B3verE
// CHECK-SAME: section "llvm.metadata"

//--- storage.cpp
// Storage-duration and scope handling for -mloadtime-comment-vars=.
//
// To be preserved a variable must have static storage duration and be defined
// at file, namespace, or class scope. A thread_local variable (thread storage
// duration) is diagnosed. A function-local static has static storage duration
// but is emitted through a different path, so it is silently ignored.
//
// Mangled names used here:
//   keep    -> keep         (namespace-scope, external linkage) -- preserved
//   N::tl   -> _ZN1N2tlE    (thread_local)                      -- diagnosed
//   stl     -> _ZL3stl      (static thread_local, internal)     -- diagnosed
//   A::tm   -> _ZN1A2tmE    (thread_local static member)        -- diagnosed
//   f()::fn -> _ZZ1fvE2fn   (function-local static)             -- ignored

// Supported: namespace scope, static storage duration -> preserved.
const char *keep = "@(#) keep";

namespace N {
// Thread storage duration -> diagnosed.
thread_local const char *tl = "@(#) tl"; // expected-warning {{'tl' named in '-mloadtime-comment-vars=' does not have static storage duration and will not be preserved}}
} // namespace N

// 'static' here only changes linkage; the storage duration is still thread.
static thread_local const char *stl = "@(#) stl"; // expected-warning {{'stl' named in '-mloadtime-comment-vars=' does not have static storage duration and will not be preserved}}

struct A {
  static thread_local const char *tm;
};
thread_local const char *A::tm = "@(#) tm"; // expected-warning {{'tm' named in '-mloadtime-comment-vars=' does not have static storage duration and will not be preserved}}

// Function-local static: static storage duration, but not emitted through the
// global-variable path, so it is silently ignored (no diagnostic, not marked).
void f() { static const char *fn = "@(#) fn"; (void)fn; }

// Only the namespace-scope variable is preserved.
// CHECK: @keep = {{.*}}!loadtime_comment
// CHECK-NOT: @_ZN1N2tlE = {{.*}}!loadtime_comment
// CHECK-NOT: @_ZL3stl = {{.*}}!loadtime_comment
// CHECK-NOT: @_ZN1A2tmE = {{.*}}!loadtime_comment
// CHECK-NOT: @_ZZ1fvE2fn = {{.*}}!loadtime_comment

//--- diag.c
// Variables named in -mloadtime-comment-vars= that the feature cannot honor are
// diagnosed, while a valid const character array is still preserved.

// Volatile-qualified pointer.
char *volatile vol_ptr = "@(#) vol ptr"; // expected-warning {{'vol_ptr' named in '-mloadtime-comment-vars=' is volatile-qualified and will not be preserved}}

// Pointer to volatile character.
volatile char *vol_char = "@(#) vol char"; // expected-warning {{'vol_char' named in '-mloadtime-comment-vars=' is volatile-qualified and will not be preserved}}

// Volatile character array.
volatile char vol_arr[] = "@(#) vol arr"; // expected-warning {{'vol_arr' named in '-mloadtime-comment-vars=' is volatile-qualified and will not be preserved}}

// Thread-local variable: does not have static storage duration.
__thread char *tls_ptr = "@(#) tls"; // expected-warning {{'tls_ptr' named in '-mloadtime-comment-vars=' does not have static storage duration and will not be preserved}}

// Pointer bound to another object (a "deferred pointer chain") rather than a
// string literal.
static const char target[] = "@(#) target";
const char *ind_ptr = target; // expected-warning {{'ind_ptr' named in '-mloadtime-comment-vars=' is not initialized with a string literal and will not be preserved}}

// A const character array is a valid form and is preserved.
const char const_arr[] = "@(#) const arr";

// The diagnosed variables are still emitted, but without the metadata that
// marks them for preservation.
// CHECK-NOT: @vol_ptr = {{.*}}!loadtime_comment
// CHECK-NOT: @vol_char = {{.*}}!loadtime_comment
// CHECK-NOT: @vol_arr = {{.*}}!loadtime_comment
// CHECK-NOT: @tls_ptr = {{.*}}!loadtime_comment
// CHECK-NOT: @ind_ptr = {{.*}}!loadtime_comment
// CHECK: @const_arr = {{.*}}constant {{.*}}!loadtime_comment

// Only const_arr is kept alive. The diagnosed variables -- including the
// deferred pointer ind_ptr -- are absent from llvm.compiler.used, so they are
// dropped from the final binary.
// CHECK: @llvm.compiler.used = appending global [1 x ptr]
// CHECK-SAME: @const_arr
// CHECK-SAME: section "llvm.metadata"

//--- init.cpp
// Initializer-form requirements for -mloadtime-comment-vars=:
//   * the variable must be constant-initialized (no dynamic initialization), so
//     that the string is present in the object at load time, and
//   * the pointer form must be bound directly to a string literal.

const char *make();

// Supported: a pointer bound to a string literal, and an array initialized
// from a string literal.
const char *p_ok = "@(#) p_ok";
char arr_ok[] = "@(#) arr_ok";

// A constant character array, referenced by a pointer below.
const char src[] = "@(#) src";

// Dynamic initialization: the value is assigned by a startup constructor, so
// the object would not contain the intended string.
const char *p_dyn = make(); // expected-warning {{'p_dyn' named in '-mloadtime-comment-vars=' is not constant-initialized and will not be preserved}}

// Constant-initialized, but the pointer is bound to another global (a "deferred
// pointer chain") rather than a string literal.
const char *p_ind = src; // expected-warning {{'p_ind' named in '-mloadtime-comment-vars=' is not initialized with a string literal and will not be preserved}}

// CHECK: @p_ok = {{.*}}!loadtime_comment
// CHECK: @arr_ok = {{.*}}!loadtime_comment
// CHECK-NOT: @p_dyn = {{.*}}!loadtime_comment
// CHECK-NOT: @p_ind = {{.*}}!loadtime_comment

// Only the two valid forms are kept alive. The dynamically initialized pointer
// and the deferred (indirect) pointer are absent from llvm.compiler.used, so
// they are dropped from the final binary rather than preserved.
// CHECK: @llvm.compiler.used = appending global [2 x ptr]
// CHECK-SAME: @p_ok
// CHECK-SAME: @arr_ok
// CHECK-SAME: section "llvm.metadata"

//--- list.c
// List-parsing edge cases.
//
// "foo, bar": the list is split at commas without trimming whitespace, so the
// second entry is " bar", which matches no mangled name. Like any other
// unrecognised name it is silently ignored: bar is emitted normally but is
// not preserved.
//
// "foo,foo": a name repeated in the list preserves the variable once; the
// duplicate entry has no additional effect.

char foo[] = "@(#) foo";
char bar[] = "@(#) bar";

void f() {}

// SPACE-DAG: @foo = global [9 x i8] c"@(#) foo\00", align {{[0-9]+}}, !loadtime_comment !{{[0-9]+}}
// SPACE-DAG: @bar = global [9 x i8] c"@(#) bar\00", align {{[0-9]+}}{{$}}
// SPACE-DAG: @llvm.compiler.used = appending global [1 x ptr] [ptr @foo], section "llvm.metadata"

// DUP-DAG: @foo = global [9 x i8] c"@(#) foo\00", align {{[0-9]+}}, !loadtime_comment !{{[0-9]+}}
// DUP-DAG: @bar = global [9 x i8] c"@(#) bar\00", align {{[0-9]+}}{{$}}
// DUP-DAG: @llvm.compiler.used = appending global [1 x ptr] [ptr @foo], section "llvm.metadata"
