// Test -mloadtime-comment-vars= IR output for C++ on AIX. Three scenarios
// are covered, each with its own set of named variables and check prefix:
//
//  CHECK     — mangled-name matching: file- and namespace-scope variables
//              receive !loadtime_comment metadata and are added to
//              llvm.compiler.used; name-matched static data members are
//              diagnosed and left unpreserved (still emitted as ordinary
//              definitions). A second pass (NOEMIT) over the same output
//              proves that listed variables of non-plain-char element type
//              (wchar_t, char16_t) are silently ignored and not emitted.
//
//  STORAGE   — storage-duration filtering: thread_local variables are
//              diagnosed by Sema and receive no metadata; a function-local
//              static is diagnosed and receives no metadata.
//
//  SPACE/DUP — list-parsing edge cases: a name with a leading space matches
//              nothing; a duplicate name preserves the variable exactly once.
//
// Names used in the matching scenario:
//
//   Source        IR symbol        Expected treatment
//   ------        ---------        ------------------
//   x             x                preserved (no mangling at C++ file scope)
//   N::x          _ZN1N1xE         preserved
//   N::ptr        _ZN1NL3ptrE      preserved ('static' gives internal linkage)
//   A::x          _ZN1A1xE         static data member: diagnosed, unpreserved
//   B::ver        _ZN1B3verE       static data member: diagnosed, unpreserved
//   C::info       _ZN1C4infoE      no definition in this TU: skipped
//   wstr          _ZL4wstr         wchar_t element type: ignored, not emitted
//   u16str        _ZL6u16str       char16_t element type: ignored, not emitted
//   sccsid_ce     _ZL9sccsid_ce    preserved (static constexpr, internal)
//   sccsid_ci     sccsid_ci        preserved (constinit; needs -std=c++20)
//   sccsid_inl    sccsid_inl       preserved (inline variable, linkonce_odr)
//
// Names used in the storage scenario (namespaces and structs are renamed to
// avoid redefinition against the matching-scenario symbols):
//
//   Source        IR symbol        Expected treatment
//   ------        ---------        ------------------
//   keep          keep             preserved (file scope, static duration)
//   S::tl         _ZN1S2tlE        thread_local: diagnosed, no metadata
//   stl           _ZL3stl          static thread_local: diagnosed, no metadata
//   T::tm         _ZN1T2tmE        thread_local static data member: diagnosed
//   g()::fn       _ZZ1gvE2fn       function-local static: diagnosed, no metadata

// RUN: %clang_cc1 -std=c++20 -O2 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=x,_ZN1N1xE,_ZN1NL3ptrE,_ZN1A1xE,_ZN1B3verE,_ZN1C4infoE,_ZL4wstr,_ZL6u16str,_ZL9sccsid_ce,sccsid_ci,sccsid_inl \
// RUN:   -emit-llvm -disable-llvm-passes -o %t.ll %s
// RUN: FileCheck %s < %t.ll
// RUN: FileCheck %s --check-prefix=NOEMIT < %t.ll

// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=keep,_ZN1S2tlE,_ZL3stl,_ZN1T2tmE,_ZZ1gvE2fn \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=STORAGE

// RUN: %clang_cc1 -std=c++20 -O2 -triple powerpc64-ibm-aix \
// RUN:   "-mloadtime-comment-vars=foo, bar" \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=SPACE

// RUN: %clang_cc1 -std=c++20 -O2 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=foo,foo \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=DUP

// ===========================================================================
// Mangled-name matching
// ===========================================================================

// 1. A file-scope array is not mangled in C++ (the IR name equals the
//    source name) and is preserved.
char x[] = "@(#) global x";

namespace N {
// 2. The namespace member N::x is matched by its mangled name _ZN1N1xE and
//    is preserved.
char x[] = "@(#) ns x";

// 3. A namespace-scope pointer initialized with a string literal is
//    preserved. The 'static' gives it internal linkage, so it mangles as
//    _ZN1NL3ptrE.
static const char *ptr = "@(#) ns ptr";
} // namespace N

// 4. The static data member A::x (mangled _ZN1A1xE) is not supported: Sema
//    diagnoses it and it gets no metadata and no compiler.used entry, though
//    it is still emitted normally as an ordinary external definition.
struct A {
  static const char *x;
};
const char *A::x = "@(#) class x";

// 5. The static data member B::ver receives the same treatment as A::x.
struct B {
  static const char *ver;
};
const char *B::ver = "@(#) class ver";

// 6. C::info is in the list but has no definition in this translation unit,
//    so it is silently skipped.
struct C { static const char *info; };

// 7. An int has an unsupported type and must not be tagged, even though its
//    IR name matches a listed name.
int not_string = 7;

// 8. These are listed, but their element type is not plain char, so they are
//    silently ignored and, being unreferenced internal-linkage statics, not
//    emitted at all.
static wchar_t wstr[] = L"@(#) wide";
static char16_t u16str[] = u"@(#) u16";

// 9. Eligible C++ declaration forms. Uses of these constant-fold, so without
//    the option none of them would be emitted at all — the forced emission is
//    what materializes the string in the object.
static constexpr const char *sccsid_ce = "@(#) constexpr";
constinit const char *sccsid_ci = "@(#) constinit";
inline const char *sccsid_inl = "@(#) inline";

void f() {}

// ===========================================================================
// Storage-duration filtering (STORAGE)
// ===========================================================================

// 10. A file-scope pointer with static storage duration is preserved.
const char *keep = "@(#) keep";

namespace S {
// 11. A thread_local variable (N renamed to S to avoid redefinition) is
//     diagnosed and receives no metadata.
thread_local const char *tl = "@(#) tl";
} // namespace S

// 12. The 'static' specifier changes linkage only; the storage duration is
//     still thread, so this is diagnosed as well.
static thread_local const char *stl = "@(#) stl";

// 13. A thread_local static data member (A renamed to T) is diagnosed and
//     receives no metadata.
struct T {
  static thread_local const char *tm;
};
thread_local const char *T::tm = "@(#) tm";

// 14. Function-local static (f renamed to g) — name-matched, so diagnosed by
//     Sema (see the Sema tests); receives no metadata either way.
void g() { static const char *fn = "@(#) fn"; (void)fn; }

// ===========================================================================
// Sources — list-parsing edge cases (SPACE, DUP)
// ===========================================================================

// 15. Simple arrays used only by the SPACE/DUP checks.
char foo[] = "@(#) foo";
char bar[] = "@(#) bar";

// ===========================================================================
// CHECK patterns — mangled-name matching
// ===========================================================================

// File-scope x and namespace N::x both matched.
// CHECK-DAG: @x = global [14 x i8] c"@(#) global x\00", align {{[0-9]+}}, !loadtime_comment ![[MD:[0-9]+]]
// CHECK-DAG: @_ZN1N1xE = global [10 x i8] c"@(#) ns x\00", align {{[0-9]+}}, !loadtime_comment ![[MD]]

// N::ptr (_ZN1NL3ptrE) points to a string literal.
// CHECK-DAG: @_ZN1NL3ptrE = internal global ptr @[[NPTR_STR:.*]], align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @[[NPTR_STR]] = private unnamed_addr constant [{{[0-9]+}} x i8] c"@(#) ns ptr\00", align {{[0-9]+}}

// A::x and B::ver are name-matched static data members: emitted as plain
// external definitions with no !loadtime_comment (the {{$}} anchors prove
// no metadata attachment).
// CHECK-DAG: @_ZN1A1xE = global ptr @{{.*}}, align {{[0-9]+}}{{$}}
// CHECK-DAG: @_ZN1B3verE = global ptr @{{.*}}, align {{[0-9]+}}{{$}}

// Invalid type must not be tagged.
// CHECK-NOT: @not_string{{.*}}!loadtime_comment

// C::info has no definition — must not appear.
// CHECK-NOT: @_ZN1C4infoE

// Eligible C++ forms: static constexpr (internal, constant), constinit
// (external), and inline (linkonce_odr) are all preserved.
// CHECK-DAG: @_ZL9sccsid_ce = internal constant ptr @[[CE_STR:.*]], align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @[[CE_STR]] = private unnamed_addr constant [15 x i8] c"@(#) constexpr\00", align {{[0-9]+}}
// CHECK-DAG: @sccsid_ci = global ptr @[[CI_STR:.*]], align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @[[CI_STR]] = private unnamed_addr constant [15 x i8] c"@(#) constinit\00", align {{[0-9]+}}
// CHECK-DAG: @sccsid_inl = linkonce_odr global ptr @[[INL_STR:.*]], align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @[[INL_STR]] = private unnamed_addr constant [12 x i8] c"@(#) inline\00", align {{[0-9]+}}

// The six supported matched globals are preserved in llvm.compiler.used;
// the two static data members are not.
// CHECK: @llvm.compiler.used = appending global [6 x ptr]
// CHECK-SAME: @x
// CHECK-SAME: @_ZN1N1xE
// CHECK-SAME: @_ZN1NL3ptrE
// CHECK-SAME: @_ZL9sccsid_ce
// CHECK-SAME: @sccsid_ci
// CHECK-SAME: @sccsid_inl
// CHECK-SAME: section "llvm.metadata"

// ===========================================================================
// NOEMIT patterns — listed variables of non-plain-char element type
// are silently ignored and, being unreferenced, not emitted at all.
// ===========================================================================

// NOEMIT-NOT: @_ZL4wstr
// NOEMIT-NOT: @_ZL6u16str

// ===========================================================================
// STORAGE patterns — thread_local variables get no metadata
// ===========================================================================

// Only the file-scope static-duration variable is preserved: it has metadata
// and appears in llvm.compiler.used. The thread_local variables are diagnosed
// by Sema and receive neither.
// STORAGE: @keep = {{.*}}!loadtime_comment
// STORAGE-NOT: @_ZN1S2tlE = {{.*}}!loadtime_comment
// STORAGE-NOT: @_ZL3stl = {{.*}}!loadtime_comment
// STORAGE-NOT: @_ZN1T2tmE = {{.*}}!loadtime_comment
// STORAGE-NOT: @_ZZ1gvE2fn = {{.*}}!loadtime_comment
// STORAGE: @llvm.compiler.used = appending global [1 x ptr] [ptr @keep], section "llvm.metadata"

// ===========================================================================
// SPACE/DUP patterns — list-parsing edge cases
// ===========================================================================

// "foo, bar": leading space means ' bar' matches nothing; only foo is preserved.
// SPACE-DAG: @foo = global [9 x i8] c"@(#) foo\00", align {{[0-9]+}}, !loadtime_comment !{{[0-9]+}}
// SPACE-DAG: @bar = global [9 x i8] c"@(#) bar\00", align {{[0-9]+}}{{$}}
// SPACE-DAG: @llvm.compiler.used = appending global [1 x ptr] [ptr @foo], section "llvm.metadata"

// "foo,foo": duplicate name preserves the variable exactly once.
// DUP-DAG: @foo = global [9 x i8] c"@(#) foo\00", align {{[0-9]+}}, !loadtime_comment !{{[0-9]+}}
// DUP-DAG: @bar = global [9 x i8] c"@(#) bar\00", align {{[0-9]+}}{{$}}
// DUP-DAG: @llvm.compiler.used = appending global [1 x ptr] [ptr @foo], section "llvm.metadata"
