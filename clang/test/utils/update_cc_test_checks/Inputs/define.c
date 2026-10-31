// Check that DEFINE and REDEFINE substitutions are expanded in RUN lines.
// DEFINE: %{optimize} = opt -passes=mem2reg \
// DEFINE:   -S
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-O0-optnone -emit-llvm %s -o - | %{optimize} | FileCheck %s --check-prefix=MEM2REG
// REDEFINE: %{optimize} = opt -passes=mem2reg,instcombine -S
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-O0-optnone -emit-llvm %s -o - | %{optimize} | FileCheck %s --check-prefix=INSTCOMBINE

int f(int a) { return (a + 1) - 1; }
