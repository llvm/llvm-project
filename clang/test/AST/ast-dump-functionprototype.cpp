// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++17 -Wno-dynamic-exception-spec -ast-dump %s | FileCheck -strict-whitespace %s

struct A {};
struct B {};

typedef void (type1)() noexcept(10 > 5);

// CHECK:      TypedefDecl {{.*}} type1 'void () noexcept(10 > 5)':'void () noexcept(10 > 5)'
// CHECK-NEXT: `-ParenType {{.*}}
// CHECK-NEXT:   `-FunctionProtoType {{.*}} 'void () noexcept(10 > 5)' exceptionspec_noexcept_true cdecl
// CHECK-NEXT:     |-NoexceptExpr: ConstantExpr {{.*}} 'bool'
// CHECK-NEXT:     | `-value: Int 1
// CHECK-NEXT:     `-BuiltinType {{.*}} 'void'

typedef void (type2)() throw(A, B);

// CHECK:      TypedefDecl {{.*}} type2 'void () throw(A, B)':'void () throw(A, B)'
// CHECK-NEXT: `-ParenType {{.*}}
// CHECK-NEXT:   `-FunctionProtoType {{.*}} 'void () throw(A, B)' exceptionspec_dynamic cdecl
// CHECK-NEXT:     |-Exceptions: 'A':'A', 'B':'B'
// CHECK-NEXT:     `-BuiltinType {{.*}} 'void'

