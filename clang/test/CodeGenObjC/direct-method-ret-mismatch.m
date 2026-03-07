// RUN: %clang_cc1 -emit-llvm -fobjc-arc -triple x86_64-apple-darwin10 %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fobjc-arc -triple arm64-apple-darwin10 -fobjc-direct-precondition-thunk %s -o - | FileCheck %s --check-prefix=EXPOSE-DIRECT

__attribute__((objc_root_class))
@interface Root
- (Root *)method __attribute__((objc_direct));
@end

// EXPOSE-DIRECT-LABEL: define ptr @useMethod
Root* useMethod(Root *root) {
  // EXPOSE-DIRECT: call ptr @"-[Root method]_thunk"
  return [root method];
}

@implementation Root
// CHECK-LABEL: define internal ptr @"\01-[Root something]"(
// EXPOSE-DIRECT-LABEL: define internal ptr @"\01-[Root something]"(ptr noundef
- (id)something {
  // CHECK: %{{[^ ]*}} = call {{.*}} @"\01-[Root method]"
  return [self method];
}

// CHECK-LABEL: define hidden ptr @"\01-[Root method]"(
// EXPOSE-DIRECT-LABEL: define hidden ptr @"-[Root method]"(ptr noundef
- (id)method {
  return self;
}

@end

// New thunk will be emitted after [Root method] instead of useMethod because its been updatd with method.
// EXPOSE-DIRECT-LABEL: define linkonce_odr hidden ptr @"-[Root method]_thunk"
// EXPOSE-DIRECT-LABEL: entry:
// EXPOSE-DIRECT:           %[[IS_NIL:.*]] = icmp eq ptr {{.*}}, null
// EXPOSE-DIRECT:           br i1 %[[IS_NIL]], label %objc_direct_method.self_is_nil, label %objc_direct_method.cont
// EXPOSE-DIRECT-LABEL: objc_direct_method.self_is_nil:
// EXPOSE-DIRECT:           call void @llvm.memset.p0.i64
// EXPOSE-DIRECT:           br label %dummy_ret_block
// EXPOSE-DIRECT-LABEL: objc_direct_method.cont:
// EXPOSE-DIRECT:           %[[RET:.*]] = musttail call ptr @"-[Root method]"
// EXPOSE-DIRECT:           ret ptr %[[RET]]
// EXPOSE-DIRECT-LABEL: dummy_ret_block:
