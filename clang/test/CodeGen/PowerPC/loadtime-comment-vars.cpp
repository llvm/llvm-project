// Test -mloadtime-comment-vars= IR output for C++-specific forms on AIX.
// Behaviour that is not specific to C++ (storage-duration filtering, list
// parsing, the option given more than once) is covered by the C tests.
//
//  CHECK     — mangled-name matching: file- and namespace-scope variables
//              receive !loadtime_comment metadata and are added to
//              llvm.compiler.used; name-matched static data members are
//              diagnosed and left unpreserved (still emitted as ordinary
//              definitions). A second pass (NOEMIT) over the same output
//              proves that listed variables of non-plain-char element type
//              (wchar_t, char16_t, char8_t) and of non-character type are
//              not preserved and not emitted. Their diagnostic is covered
//              by the Sema tests and silenced here. The names are split
//              across two occurrences of the option, which are combined.
//
// Names used:
//
//   Source        IR symbol                Expected treatment
//   ------        ---------                ------------------
//   x             x                        preserved (no mangling at C++ file scope)
//   N::x          _ZN1N1xE                 preserved
//   N::ptr        _ZN1NL3ptrE              preserved ('static' gives internal linkage)
//   A::x          _ZN1A1xE                 static data member: diagnosed, unpreserved
//   B::ver        _ZN1B3verE               static data member: diagnosed, unpreserved
//   C::info       _ZN1C4infoE              no definition in this TU: skipped
//   wstr          _ZL4wstr                 wchar_t element type: diagnosed, not emitted
//   u16str        _ZL6u16str               char16_t element type: diagnosed, not emitted
//   u8str         _ZL5u8str                char8_t element type: diagnosed, not emitted
//   not_string    not_string               int: unsupported type: diagnosed, not emitted
//   cver          cver                     preserved (C linkage, not mangled)
//   anon          _ZN12_GLOBAL__N_14anonE  preserved (unnamed namespace)
//   version       asmid                    preserved (asm label)
//   sccsid_ce     _ZL9sccsid_ce            preserved (static constexpr, internal)
//   sccsid_ci     sccsid_ci                preserved (constinit; needs -std=c++20)
//   sccsid_br     _ZL9sccsid_br            preserved (braced string literal)
//   [a, b, c]     _ZDC1a1b1cE              preserved (structured binding: the
//                                          DecompositionDecl owns the storage)

// RUN: %clang_cc1 -std=c++20 -O2 -triple powerpc64-ibm-aix -Wno-loadtime-comment-var \
// RUN:   -mloadtime-comment-vars=x,_ZN1N1xE,_ZN1NL3ptrE,_ZN1A1xE,_ZN1B3verE,_ZN1C4infoE,not_string,_ZL4wstr \
// RUN:   -mloadtime-comment-vars=_ZL6u16str,_ZL5u8str,_ZL9sccsid_ce,sccsid_ci,_ZL9sccsid_br,_ZDC1a1b1cE,cver,_ZN12_GLOBAL__N_14anonE,asmid \
// RUN:   -emit-llvm -disable-llvm-passes -o %t.ll %s
// RUN: FileCheck %s < %t.ll
// RUN: FileCheck %s --check-prefix=NOEMIT < %t.ll

// ===========================================================================
// Sources
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

// 7. An int has an unsupported type: it is diagnosed (see the Sema tests) and
//    must not be tagged, even though its IR name matches a listed name.
int not_string = 7;

// 8. These are listed, but their element type is not plain char, so they are
//    diagnosed (see the Sema tests), not preserved, and -- being unreferenced
//    internal-linkage statics -- not emitted at all.
static wchar_t wstr[] = L"@(#) wide";
static char16_t u16str[] = u"@(#) u16";
static char8_t u8str[] = u8"@(#) u8";

// 9. Eligible C++ declaration forms. Uses of these constant-fold, so without
//    the option none of them would be emitted at all — the forced emission is
//    what materializes the string in the object.
static constexpr const char *sccsid_ce = "@(#) constexpr";
constinit const char *sccsid_ci = "@(#) constinit";
static const char *sccsid_br{"@(#) braced"};

// 10. Structured binding. The bindings a/b/c are BindingDecls with no storage
//     of their own; the hidden DecompositionDecl (a VarDecl) owns the array,
//     and the symbol is the mangled name of that VarDecl, so it is the
//     DecompositionDecl that is preserved.
auto [a, b, c] = "ab";

void f() {}

// 11. C language linkage: the name is not mangled, so the identifier is the
//     IR name and the listed name.
extern "C" char cver[] = "@(#) c linkage";

// 12. An unnamed namespace mangles with the _GLOBAL__N_1 marker.
namespace {
char anon[] = "@(#) anon";
} // namespace

// 13. An asm label replaces the mangled name in the object file, so the
//     variable is listed by its label.
char version[] asm("asmid") = "@(#) asm label";

// ===========================================================================
// CHECK patterns
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

// CHECK-DAG: @cver = global [15 x i8] c"@(#) c linkage\00", align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @_ZN12_GLOBAL__N_14anonE = internal global [10 x i8] c"@(#) anon\00", align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @asmid = global [15 x i8] c"@(#) asm label\00", align {{[0-9]+}}, !loadtime_comment ![[MD]]

// Eligible C++ forms: static constexpr (internal, constant), constinit
// (external), and a pointer list-initialized from a braced string literal
// are all preserved.
// CHECK-DAG: @_ZL9sccsid_ce = internal constant ptr @[[CE_STR:.*]], align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @[[CE_STR]] = private unnamed_addr constant [15 x i8] c"@(#) constexpr\00", align {{[0-9]+}}
// CHECK-DAG: @sccsid_ci = global ptr @[[CI_STR:.*]], align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @[[CI_STR]] = private unnamed_addr constant [15 x i8] c"@(#) constinit\00", align {{[0-9]+}}
// CHECK-DAG: @_ZL9sccsid_br = internal global ptr @[[BR_STR:.*]], align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @[[BR_STR]] = private unnamed_addr constant [12 x i8] c"@(#) braced\00", align {{[0-9]+}}
// CHECK-DAG: @_ZDC1a1b1cE = internal constant [3 x i8] c"ab\00", align {{[0-9]+}}, !loadtime_comment ![[MD]]

// The ten supported matched globals are preserved in llvm.compiler.used;
// the two static data members are not.
// CHECK: @llvm.compiler.used = appending global [10 x ptr]
// CHECK-SAME: @x
// CHECK-SAME: @_ZN1N1xE
// CHECK-SAME: @_ZN1NL3ptrE
// CHECK-SAME: @_ZL9sccsid_ce
// CHECK-SAME: @sccsid_ci
// CHECK-SAME: @_ZL9sccsid_br
// CHECK-SAME: @_ZDC1a1b1cE
// CHECK-SAME: @cver
// CHECK-SAME: @_ZN12_GLOBAL__N_14anonE
// CHECK-SAME: @asmid
// CHECK-SAME: section "llvm.metadata"

// ===========================================================================
// NOEMIT patterns — listed variables of non-plain-char element type are
// diagnosed, not preserved, and, being unreferenced, not emitted at all.
// ===========================================================================

// NOEMIT-NOT: @_ZL4wstr
// NOEMIT-NOT: @_ZL6u16str
// NOEMIT-NOT: @_ZL5u8str
