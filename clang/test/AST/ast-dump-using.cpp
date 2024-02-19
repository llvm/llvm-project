// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump %s | FileCheck -strict-whitespace %s

namespace a {
struct S;
}
namespace b {
using a::S;
// CHECK:      UsingDecl {{.*}} a::S
// CHECK-NEXT: | `-NestedNameSpecifier Namespace {{.*}} 'a'
// CHECK-NEXT: UsingShadowDecl {{.*}} implicit CXXRecord {{.*}} 'S'
// CHECK-NEXT: `-RecordType {{.*}} 'a::S'
typedef S f; // to dump the introduced type
// CHECK:      TypedefDecl
// CHECK-NEXT: `-ElaboratedType {{.*}} 'S' sugar
// CHECK-NEXT:   `-UsingType [[TYPE_ADDR:.*]] 'a::S' sugar
// CHECK-NEXT:     |-UsingShadow [[SHADOW_ADDR:.*]] 'S'
// CHECK-NEXT:     `-RecordType {{.*}} 'a::S'
typedef S e; // check the same UsingType is reused.
// CHECK:      TypedefDecl
// CHECK-NEXT: `-ElaboratedType {{.*}} 'S' sugar
// CHECK-NEXT:   `-UsingType [[TYPE_ADDR]] 'a::S' sugar
// CHECK-NEXT:     |-UsingShadow [[SHADOW_ADDR]] 'S'
// CHECK-NEXT:     `-RecordType {{.*}} 'a::S'
}
