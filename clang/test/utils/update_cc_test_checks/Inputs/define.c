// Check that DEFINE and REDEFINE substitutions are expanded in RUN lines.
// DEFINE: %{optimize} = opt -passes=mem2reg \
// DEFINE:   -S
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-O0-optnone -emit-llvm %s -o - | %{optimize} | FileCheck %s --check-prefix=MEM2REG
// REDEFINE: %{optimize} = opt -passes=mem2reg,instcombine -S
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-O0-optnone -emit-llvm %s -o - | %{optimize} | FileCheck %s --check-prefix=INSTCOMBINE
// DEFINE: %{pass1} = mem2reg
// DEFINE: %{pass2} = instcombine
// DEFINE: %{optimize-nested} = opt -passes=%{pass1},%{pass2} -S
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-O0-optnone -emit-llvm %s -o - | %{optimize-nested} | FileCheck %s --check-prefix=NESTED

int f(int a) { return (a + 1) - 1; }
