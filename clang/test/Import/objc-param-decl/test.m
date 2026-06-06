// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// RUN: clang-import-test -dump-ast -x objective-c++ -import %S/Inputs/S.m -expression %s | FileCheck %s

// CHECK: ObjCTypeParamDecl
// CHECK-SAME: FirstParam
// CHECK-SAME: 'id<NSString>'
// CHECK-NEXT: ObjCTypeParamDecl
// CHECK-SAME: 'id'

void expr() {
  Dictionary *d;
}
