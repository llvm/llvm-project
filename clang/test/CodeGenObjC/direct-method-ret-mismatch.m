// RUN: %clang_cc1 -emit-llvm -fobjc-arc -triple x86_64-apple-darwin10 %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fobjc-arc -triple arm64-apple-darwin10 -fobjc-direct-precondition-thunk %s -o - | FileCheck %s --check-prefix=EXPOSE-DIRECT

__attribute__((objc_root_class))
@interface Root
- (Root *)method __attribute__((objc_direct));
@end

// EXPOSE-DIRECT-LABEL: define ptr @useMethod
Root* useMethod(Root *root) {
  // EXPOSE-DIRECT: call ptr @"-[Root method]D_thunk"
  return [root method];
}

@implementation Root
// The IR emission order depends on the ret-type mismatch: [Root method] is
// emitted before [Root something] because GenerateDirectMethod replaces the
// function when declaration (Root *) and implementation (id) types differ.
//
// CHECK-LABEL: define hidden ptr @"\01-[Root method]"(
// EXPOSE-DIRECT-LABEL: define hidden ptr @"-[Root method]D"(ptr noundef
- (id)method {
  return self;
}

// EXPOSE-DIRECT-LABEL: define linkonce_odr hidden ptr @"-[Root method]D_thunk"
// EXPOSE-DIRECT-LABEL: entry:
// EXPOSE-DIRECT:           %[[IS_NIL:.*]] = icmp eq ptr {{.*}}, null
// EXPOSE-DIRECT:           br i1 %[[IS_NIL]], label %objc_direct_method.self_is_nil, label %objc_direct_method.cont
// EXPOSE-DIRECT-LABEL: objc_direct_method.self_is_nil:
// EXPOSE-DIRECT:           call void @llvm.memset.p0.i64
// EXPOSE-DIRECT:           br label %dummy_ret_block
// EXPOSE-DIRECT-LABEL: objc_direct_method.cont:
// EXPOSE-DIRECT:           %[[RET:.*]] = musttail call ptr @"-[Root method]D"
// EXPOSE-DIRECT:           ret ptr %[[RET]]
// EXPOSE-DIRECT-LABEL: dummy_ret_block:

// CHECK-LABEL: define internal ptr @"\01-[Root something]"(
// EXPOSE-DIRECT-LABEL: define internal ptr @"\01-[Root something]"(ptr noundef
- (id)something {
  // CHECK: %{{[^ ]*}} = call {{.*}} @"\01-[Root method]"
  return [self method];
}

@end
