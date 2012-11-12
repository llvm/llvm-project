// RUN: %clang_cc1 -emit-llvm < %s -o - | FileCheck %s

void a() { __builtin_unwind_init(); }

// CHECK:  call void @llvm.eh.unwind.init()
