// RUN: %clang_cc1 -std=c++20 %s -ast-dump | FileCheck %s
export module a;
export class f {
public:
    void non_inline_func() {}
    constexpr void constexpr_func() {}
    consteval void consteval_func() {}
};

// CHECK-NOT: non_inline_func {{.*}}implicit-inline
// CHECK: constexpr_func {{.*}}implicit-inline
// CHECK: consteval_func {{.*}}implicit-inline
