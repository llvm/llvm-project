// RUN: %clang_cc1 -fms-compatibility -fms-compatibility-version=19.33 -std=c++20 -ast-dump -verify %s | FileCheck %s
// RUN: %clang_cc1 -fms-compatibility -fms-compatibility-version=19.33 -std=c++20 -ast-dump -verify %s -fexperimental-new-constant-interpreter | FileCheck %s
// expected-no-diagnostics

// CHECK: used f1 'bool ()'
// CHECK: MSConstexprAttr 0x{{[0-9a-f]+}} <col:3, col:9>
[[msvc::constexpr]] bool f1() { return true; }

// CHECK: used constexpr f2 'bool ()'
// CHECK-NEXT: CompoundStmt 0x{{[0-9a-f]+}} <col:21, col:56>
// CHECK-NEXT: AttributedStmt 0x{{[0-9a-f]+}} <col:23, col:53>
// CHECK-NEXT: MSConstexprAttr 0x{{[0-9a-f]+}} <col:25, col:31>
// CHECK-NEXT: ReturnStmt 0x{{[0-9a-f]+}} <col:43, col:53>
constexpr bool f2() { [[msvc::constexpr]] return f1(); }
static_assert(f2());

struct S1 {
    // CHECK: used vm 'bool ()' virtual
    // CHECK: MSConstexprAttr 0x{{[0-9a-f]+}} <col:7, col:13>
    [[msvc::constexpr]] virtual bool vm() { return true; }

    // CHECK: used constexpr cm 'bool ()'
    // CHECK-NEXT: CompoundStmt 0x{{[0-9a-f]+}} <col:25, col:60>
    // CHECK-NEXT: AttributedStmt 0x{{[0-9a-f]+}} <col:27, col:57>
    // CHECK-NEXT: MSConstexprAttr 0x{{[0-9a-f]+}} <col:29, col:35>
    // CHECK-NEXT: ReturnStmt 0x{{[0-9a-f]+}} <col:47, col:57>
    constexpr bool cm() { [[msvc::constexpr]] return vm(); }
};
static_assert(S1{}.cm());
