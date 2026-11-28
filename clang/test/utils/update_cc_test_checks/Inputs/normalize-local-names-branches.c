// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -discard-value-names -disable-O0-optnone -o - %s | opt -passes=mem2reg -S | FileCheck %s

int select(int a, int b) { return a ? a : b; }
