// RUN: %clang_cc1 %s -DTEST_EMPTY_COPYRIGHT -triple powerpc-ibm-aix -verify
// RUN: %clang_cc1 %s -DTEST_OTHER_COMMENT_KINDS -triple powerpc-ibm-aix -verify
// RUN: %clang_cc1 %s -triple powerpc-ibm-aix -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple powerpc-ibm-aix -verify

// RUN: %clang_cc1 %s -DTEST_EMPTY_COPYRIGHT -triple powerpc64-ibm-aix -verify
// RUN: %clang_cc1 %s -DTEST_OTHER_COMMENT_KINDS -triple powerpc64-ibm-aix -verify
// RUN: %clang_cc1 %s -triple powerpc64-ibm-aix -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple powerpc64-ibm-aix -verify

#ifdef TEST_EMPTY_COPYRIGHT

// 1. Verify no diagnostics for empty copyright string
#pragma comment(copyright, "") // expected-no-diagnostics
int main() {return 0; }

#elif defined(TEST_OTHER_COMMENT_KINDS)

// 2. Verify warnings for lib/linker and silent ignore for others
#pragma comment(lib, "m") // expected-warning {{'#pragma comment lib' ignored}}
#pragma comment(linker, "foo") // expected-warning {{'#pragma comment linker' ignored}}

// These are recognized but ignored in CodeGen
#pragma comment(compiler) // expected-warning {{'#pragma comment compiler' ignored}}
#pragma comment(exestr, "foo") // expected-warning {{'#pragma comment exestr' ignored}}
#pragma comment(user, "foo\abar\nbaz\tsomething") // expected-warning {{'#pragma comment user' ignored}}
int main() {return 0; }

#else

// 3. Default Path: Verify metadata generation and duplicate warning
#pragma comment(copyright, "@(#) Copyright")
#pragma comment(copyright, "Duplicate Copyright") // expected-warning {{'#pragma comment copyright' ignored: it can be specified only once per translation unit}}

int main() { return 0; }
// Check that both metadata sections are present
// CHECK: !comment_string.loadtime = !{![[copyright:[0-9]+]]}

// Check individual metadata content
// CHECK: ![[copyright]] = !{!"@(#) Copyright"}

#endif
