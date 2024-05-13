// RUN: %clang_cc1 -triple i686-windows-msvc -fms-extensions -emit-llvm -std=c++1y -O0 -o - %s | FileCheck %s
struct Base {};

// __declspec(dllexport) causes us to export the implicit constructor.
struct __declspec(dllexport) Derived : virtual Base {
// CHECK-LABEL: define weak_odr dso_local dllexport x86_thiscallcc noundef ptr @"??0Derived@@QAE@ABU0@@Z"
// CHECK:      %[[this:.*]] = load ptr, ptr {{.*}}
// CHECK-NEXT: store ptr %[[this]], ptr %[[retval:.*]]
// CHECK:      %[[dest_a_gep:.*]] = getelementptr inbounds %struct.Derived, ptr %[[this]], i32 0, i32 1
// CHECK-NEXT: %[[src_load:.*]]   = load ptr, ptr {{.*}}
// CHECK-NEXT: %[[src_a_gep:.*]]  = getelementptr inbounds %struct.Derived, ptr %[[src_load:.*]], i32 0, i32 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[dest_a_gep]], ptr align 4 %[[src_a_gep]], i64 1, i1 false)
// CHECK-NEXT: %[[dest_this:.*]] = load ptr, ptr %[[retval]]
// CHECK-NEXT: ret ptr %[[dest_this]]
  bool a : 1;
  bool b : 1;
};

// __declspec(dllexport) causes us to export the implicit copy constructor.
struct __declspec(dllexport) Derived2 : virtual Base {
// CHECK-LABEL: define weak_odr dso_local dllexport x86_thiscallcc noundef ptr @"??0Derived2@@QAE@ABU0@@Z"
// CHECK:      %[[this:.*]] = load ptr, ptr {{.*}}
// CHECK-NEXT: store ptr %[[this]], ptr %[[retval:.*]]
// CHECK:      %[[dest_a_gep:.*]] = getelementptr inbounds %struct.Derived2, ptr %[[this]], i32 0, i32 1
// CHECK-NEXT: %[[src_load:.*]]   = load ptr, ptr {{.*}}
// CHECK-NEXT: %[[src_a_gep:.*]]  = getelementptr inbounds %struct.Derived2, ptr %[[src_load:.*]], i32 0, i32 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %[[dest_a_gep]], ptr align 4 %[[src_a_gep]], i32 4, i1 false)
// CHECK-NEXT: %[[dest_this:.*]] = load ptr, ptr %[[retval]]
// CHECK-NEXT: ret ptr %[[dest_this]]
  int Array[1];
};
