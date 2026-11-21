// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-O0-optnone -o - %s | opt -passes=mem2reg -S | FileCheck %s --check-prefixes=C,LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-O0-optnone -o - %s | FileCheck %s --check-prefixes=C,LLVM

int add_one(int x) { return x + 1; }
