// This test uses PrintFunctionNames with -fdelayed-template-parsing because it
// happens to use a RecursiveASTVisitor that forces deserialization of AST
// files.
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -fdelayed-template-parsing \
// RUN:      -std=c++14 -emit-pch -o %t.pch %s
// RUN: %clang_cc1 -triple %itanium_abi_triple \
// RUN:     -load %llvmshlibdir/PrintFunctionNames%pluginext \
// RUN:     -add-plugin print-fns -std=c++14 -include-pch %t.pch %s -emit-llvm \
// RUN:     -fdelayed-template-parsing -debug-info-kind=limited \
// RUN:     -o %t.ll 2>&1 | FileCheck --check-prefix=DECLS %s
// RUN: FileCheck --check-prefix=IR %s < %t.ll
//
// REQUIRES: plugins, examples

// DECLS: top-level-decl: "func"

// IR: define {{.*}}void @_Z4funcv()

#ifndef HEADER
#define HEADER

struct nullopt_t {
  constexpr explicit nullopt_t(int) {}
};
constexpr nullopt_t nullopt(0);

#else

void func() { }

#endif
