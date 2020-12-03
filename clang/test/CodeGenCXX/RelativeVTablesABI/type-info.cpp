// Check typeid() + type_info

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O3 -S -o - -emit-llvm -fcxx-exceptions -fexceptions -fexperimental-relative-c++-abi-vtables | FileCheck %s

// CHECK: %class.A = type { i32 (...)** }
// CHECK: %class.B = type { %class.A }
// CHECK: %"class.std::type_info" = type { i32 (...)**, i8* }

// CHECK: $_ZTI1A.rtti_proxy = comdat any
// CHECK: $_ZTI1B.rtti_proxy = comdat any

// CHECK: @_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
// CHECK: @_ZTS1A = constant [3 x i8] c"1A\00", align 1
// CHECK: @_ZTI1A = constant { i8*, i8* } { i8* getelementptr inbounds (i8, i8* bitcast (i8** @_ZTVN10__cxxabiv117__class_type_infoE to i8*), i32 8), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1A, i32 0, i32 0) }, align 8
// CHECK: @_ZTVN10__cxxabiv120__si_class_type_infoE = external global i8*
// CHECK: @_ZTS1B = constant [3 x i8] c"1B\00", align 1
// CHECK: @_ZTI1B = constant { i8*, i8*, i8* } { i8* getelementptr inbounds (i8, i8* bitcast (i8** @_ZTVN10__cxxabiv120__si_class_type_infoE to i8*), i32 8), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1B, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*) }, align 8
// CHECK: @_ZTI1A.rtti_proxy = hidden unnamed_addr constant { i8*, i8* }* @_ZTI1A, comdat
// CHECK: @_ZTI1B.rtti_proxy = hidden unnamed_addr constant { i8*, i8*, i8* }* @_ZTI1B, comdat

// CHECK:      define {{.*}}%"class.std::type_info"* @_Z11getTypeInfov() local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret %"class.std::type_info"* bitcast ({ i8*, i8* }* @_ZTI1A to %"class.std::type_info"*)
// CHECK-NEXT: }

// CHECK:      define i8* @_Z7getNamev() local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1A, i64 0, i64 0)
// CHECK-NEXT: }

// CHECK:      define i1 @_Z5equalP1A(%class.A* readonly %a) local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[isnull:%[0-9]+]] = icmp eq %class.A* %a, null
// CHECK-NEXT:   br i1 [[isnull]], label %[[bad_typeid:[a-z0-9._]+]], label %[[end:[a-z0-9.+]+]]
// CHECK:      [[bad_typeid]]:
// CHECK-NEXT:   tail call void @__cxa_bad_typeid()
// CHECK-NEXT:   unreachable
// CHECK:      [[end]]:
// CHECK-NEXT:   [[type_info_ptr3:%[0-9]+]] = bitcast %class.A* %a to i8**
// CHECK-NEXT:   [[vtable:%[a-z0-9]+]] = load i8*, i8** [[type_info_ptr3]]
// CHECK-NEXT:   [[type_info_ptr:%[0-9]+]] = tail call i8* @llvm.load.relative.i32(i8* [[vtable]], i32 -4)
// CHECK-NEXT:   [[type_info_ptr2:%[0-9]+]] = bitcast i8* [[type_info_ptr]] to %"class.std::type_info"**
// CHECK-NEXT:   [[type_info_ptr:%[0-9]+]] = load %"class.std::type_info"*, %"class.std::type_info"** [[type_info_ptr2]], align 8
// CHECK-NEXT:   [[name_ptr:%[a-z0-9._]+]] = getelementptr inbounds %"class.std::type_info", %"class.std::type_info"* [[type_info_ptr]], i64 0, i32 1
// CHECK-NEXT:   [[name:%[0-9]+]] = load i8*, i8** [[name_ptr]], align 8
// CHECK-NEXT:   [[eq:%[a-z0-9.]+]] = icmp eq i8* [[name]], getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1B, i64 0, i64 0)
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
