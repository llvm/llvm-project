// Check typeid() + type_info

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O3 -o - -emit-llvm -fcxx-exceptions -fexceptions | FileCheck %s

// CHECK: $_ZTI1A.rtti_proxy = comdat any
// CHECK: $_ZTI1B.rtti_proxy = comdat any

// CHECK: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// CHECK: @_ZTS1A ={{.*}} constant [3 x i8] c"1A\00", align 1
// CHECK: @_ZTI1A ={{.*}} constant { ptr, ptr } { ptr getelementptr inbounds (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i32 8), ptr @_ZTS1A }, align 8
// CHECK: @_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
// CHECK: @_ZTS1B ={{.*}} constant [3 x i8] c"1B\00", align 1
// CHECK: @_ZTI1B ={{.*}} constant { ptr, ptr, ptr } { ptr getelementptr inbounds (i8, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i32 8), ptr @_ZTS1B, ptr @_ZTI1A }, align 8
// CHECK: @_ZTI1A.rtti_proxy = linkonce_odr hidden unnamed_addr constant ptr @_ZTI1A, comdat
// CHECK: @_ZTI1B.rtti_proxy = linkonce_odr hidden unnamed_addr constant ptr @_ZTI1B, comdat

// CHECK:      define {{.*}}ptr @_Z11getTypeInfov() local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret ptr @_ZTI1A
// CHECK-NEXT: }

// CHECK:      define{{.*}} ptr @_Z7getNamev() local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret ptr @_ZTS1A
// CHECK-NEXT: }

// CHECK:      define{{.*}} i1 @_Z5equalP1A(ptr noundef readonly %a) local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[isnull:%[0-9]+]] = icmp eq ptr %a, null
// CHECK-NEXT:   br i1 [[isnull]], label %[[bad_typeid:[a-z0-9._]+]], label %[[end:[a-z0-9.+]+]]
// CHECK:      [[bad_typeid]]:
// CHECK-NEXT:   tail call void @__cxa_bad_typeid()
// CHECK-NEXT:   unreachable
// CHECK:      [[end]]:
// CHECK-NEXT:   [[vtable:%[a-z0-9]+]] = load ptr, ptr %a
// CHECK-NEXT:   [[type_info_ptr:%[0-9]+]] = tail call ptr @llvm.load.relative.i32(ptr [[vtable]], i32 -4)
// CHECK-NEXT:   [[type_info_ptr2:%[0-9]+]] = load ptr, ptr [[type_info_ptr]], align 8
// CHECK-NEXT:   [[name_ptr:%[a-z0-9._]+]] = getelementptr inbounds nuw i8, ptr [[type_info_ptr2]], i64 8
// CHECK-NEXT:   [[name:%[0-9]+]] = load ptr, ptr [[name_ptr]], align 8
// CHECK-NEXT:   [[eq:%[a-z0-9.]+]] = icmp eq ptr [[name]], @_ZTS1B
// CHECK-NEXT:   ret i1 [[eq]]
// CHECK-NEXT: }

#include "../typeinfo"

class A {
public:
  virtual void foo();
};

class B : public A {
public:
  void foo() override;
};

void A::foo() {}
void B::foo() {}

const auto &getTypeInfo() {
  return typeid(A);
}

const char *getName() {
  return typeid(A).name();
}

bool equal(A *a) {
  return typeid(B) == typeid(*a);
}
