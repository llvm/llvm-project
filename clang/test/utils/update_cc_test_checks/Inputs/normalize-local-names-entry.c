// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-O0-optnone -o - %s | opt -passes=mem2reg -S | FileCheck %s

int f(int entryv) { return entryv + 1; }
