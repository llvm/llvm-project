// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump %s | FileCheck %s

int * _Nonnull x;
using Ty = decltype(x);

// CHECK: TypeAliasDecl 0x{{[^ ]*}}  <line:4:1, col:22> col:7 Ty 'decltype(x)':'int *'
// CHECK-NEXT:  `-typeDetails: DecltypeType 0x{{[^ ]*}} 'decltype(x)' sugar
// CHECK-NEXT:     |-DeclRefExpr 0x{{[^ ]*}} <col:21> 'int * _Nonnull':'int *' lvalue Var 0x{{[^ ]*}} 'x' 'int * _Nonnull':'int *' non_odr_use_unevaluated
// CHECK-NEXT:    `-typeDetails: AttributedType 0x{{[^ ]*}} 'int * _Nonnull' sugar
// CHECK-NEXT:      `-typeDetails: PointerType 0x{{[^ ]*}} 'int *'
// CHECK-NEXT:        `-typeDetails: BuiltinType 0x{{[^ ]*}} 'int'


[[clang::address_space(3)]] int *y;
using Ty1 = decltype(y);

// CHECK: TypeAliasDecl 0x{{[^ ]*}} <line:15:1, col:23> col:7 Ty1 'decltype(y)':'__attribute__((address_space(3))) int *'
// CHECK-NEXT: `-typeDetails: DecltypeType 0x{{[^ ]*}} 'decltype(y)' sugar
// CHECK-NEXT:   |-DeclRefExpr 0x{{[^ ]*}} <col:22> '__attribute__((address_space(3))) int *' lvalue Var 0x{{[^ ]*}} 'y' '__attribute__((address_space(3))) int *' non_odr_use_unevaluated
// CHECK-NEXT:     `-typeDetails: PointerType 0x{{[^ ]*}} '__attribute__((address_space(3))) int *'
// CHECK-NEXT:       `-typeDetails: AttributedType 0x{{[^ ]*}} '__attribute__((address_space(3))) int' sugar
// CHECK-NEXT          |-typeDetails: BuiltinType 0x{{[^ ]*}} 'int'
// CHECK-NEXT          `-typeDetails: QualType 0x{{[^ ]*}} '__attribute__((address_space(3))) int' __attribute__((address_space(3)))
// CHECK-NEXT            `-typeDetails: BuiltinType 0x{{[^ ]*}} 'int'
