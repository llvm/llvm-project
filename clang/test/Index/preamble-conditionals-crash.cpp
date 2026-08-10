#ifndef HEADER_GUARD

#define FOO int aba;
FOO

#endif
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 5 \
// RUN:                                       local -std=c++14 %s 2>&1 \
// RUN: | FileCheck %s --implicit-check-not "libclang: crash detected" \
// RUN:                --implicit-check-not "error:"
// CHECK: macro expansion=FOO:3:9 Extent=[4:1 - 4:4]
// CHECK: VarDecl=aba:4:1 (Definition) Extent=[4:1 - 4:4]
