// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited < %s | grep DW_TAG_member

struct A { int x; } a;
