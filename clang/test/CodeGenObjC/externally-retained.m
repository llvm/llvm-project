// RUN: %clang_cc1 -triple x86_64-apple-macosx10.13.0 -fobjc-arc -fblocks -Wno-objc-root-class -O0 %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.13.0 -fobjc-arc -fblocks -Wno-objc-root-class -O0 -xobjective-c++ -std=c++11 %s -S -emit-llvm -o - | FileCheck %s --check-prefix CHECKXX

#define EXT_RET __attribute__((objc_externally_retained))

@interface ObjTy @end

ObjTy *global;

#if __cplusplus
// Suppress name mangling in C++ mode for the sake of check lines.
extern "C" void param(ObjTy *p);
extern "C" void local();
extern "C" void in_init();
extern "C" void anchor();
extern "C" void block_capture(ObjTy *);
extern "C" void esc(void (^)());
extern "C" void escp(void (^)(ObjTy *));
extern "C" void block_param();
#endif

void param(ObjTy *p) EXT_RET {
  // CHECK-LABEL: define{{.*}} void @param
  // CHECK-NOT: llvm.objc.
  // CHECK: ret
}

void local(void) {
  EXT_RET ObjTy *local = global;
  // CHECK-LABEL: define{{.*}} void @local
  // CHECK-NOT: llvm.objc.
  // CHECK: ret
}

void in_init(void) {
  // Test that we do the right thing when a variable appears in it's own
  // initializer. Here, we release the value stored in 'wat' after overwriting
  // it, in case it was somehow set to point to a non-null object while it's
  // initializer is being evaluated.
  EXT_RET ObjTy *wat = 0 ? wat : global;

  // CHECK-LABEL: define{{.*}} void @in_init
  // CHECK: [[WAT:%.*]] = alloca
  // CHECK-NEXT: store {{.*}} null, {{.*}} [[WAT]]
  // CHECK-NEXT: [[GLOBAL:%.*]] = load {{.*}} @global
  // CHECK-NEXT: [[WAT_LOAD:%.*]] = load {{.*}} [[WAT]]
  // CHECK-NEXT: store {{.*}} [[GLOBAL]], {{.*}} [[WAT]]
  // CHECK-NEXT: [[CASTED:%.*]] = bitcast {{.*}} [[WAT_LOAD]] to
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[CASTED]])

  // CHECK-NOT: llvm.objc.
  // CHECK: ret
}

void esc(void (^)(void));

void block_capture(ObjTy *obj) EXT_RET {
  esc(^{ (void)obj; });

  // CHECK-LABEL: define{{.*}} void @block_capture
  // CHECK-NOT: llvm.objc.
  // CHECK: call i8* @llvm.objc.retain
  // CHECK-NOT: llvm.objc.
  // CHECK: call void @esc
  // CHECK-NOT: llvm.objc.
  // CHECK: call void @llvm.objc.storeStrong({{.*}} null)
  // CHECK-NOT: llvm.objc.
  // CHECK: ret

  // CHECK-LABEL: define {{.*}} void @__copy_helper_block_
  // CHECK-NOT: llvm.objc.
  // CHECK: llvm.objc.storeStrong
  // CHECK-NOT: llvm.objc.
  // CHECK: ret

  // CHECK-LABEL: define {{.*}} void @__destroy_helper_block_
  // CHECK-NOT: llvm.objc.
  // CHECK: llvm.objc.storeStrong({{.*}} null)
  // CHECK-NOT: llvm.objc.
  // CHECK: ret
}

void escp(void (^)(ObjTy *));

void block_param(void) {
  escp(^(ObjTy *p) EXT_RET {});

  // CHECK-LABEL: define internal void @__block_param_block_invoke
  // CHECK-NOT: llvm.objc.
  // CHECK: ret
}

@interface Inter
-(void)m1: (ObjTy *)w;
@end

@implementation Inter
-(void)m1: (ObjTy *) w EXT_RET {
  // CHECK-LABEL: define internal void @"\01-[Inter m1:]"
  // CHECK-NOT: llvm.objc.
  // CHECK: ret
}
-(void)m2: (ObjTy *) w EXT_RET {
  // CHECK-LABEL: define internal void @"\01-[Inter m2:]"
  // CHECK-NOT: llvm.objc.
  // CHECK: ret
}
@end

#if __cplusplus
// Verify that the decltype(p) is resolved before 'p' is made implicitly const.
__attribute__((objc_externally_retained))
void foo(ObjTy *p, decltype(p) *) {}
// CHECKXX: _Z3fooP5ObjTyPU8__strongS0_
#endif
