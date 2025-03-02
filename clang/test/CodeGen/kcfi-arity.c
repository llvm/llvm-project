// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -fsanitize-kcfi-arity -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -fsanitize-kcfi-arity -x c++ -o - %s | FileCheck %s
#if !__has_feature(kcfi_arity)
#error Missing kcfi_arity?
#endif

// CHECK: ![[#]] = !{i32 4, !"kcfi-arity", i32 1}
