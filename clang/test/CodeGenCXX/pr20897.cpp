// RUN: %clang_cc1 -triple i686-windows-msvc -fms-extensions -emit-llvm -std=c++1y -O0 -o - %s | FileCheck %s
struct Base {};

// __declspec(dllexport) causes us to export the implicit constructor.
struct __declspec(dllexport) Derived : virtual Base {
// CHECK-LABEL: define weak_odr dso_local dllexport x86_thiscallcc noundef %struct.Derived* @"??0Derived@@QAE@ABU0@@Z"
// CHECK:      %[[this:.*]] = load %struct.Derived*, %struct.Derived** {{.*}}
// CHECK-NEXT: store %struct.Derived* %[[this]], %struct.Derived** %[[retval:.*]]
// CHECK:      %[[dest_a_gep:.*]] = getelementptr inbounds %struct.Derived, %struct.Derived* %[[this]], i32 0, i32 1
// CHECK-NEXT: %[[src_load:.*]]   = load %struct.Derived*, %struct.Derived** {{.*}}
// CHECK-NEXT: %[[src_a_gep:.*]]  = getelementptr inbounds %struct.Derived, %struct.Derived* %[[src_load:.*]], i32 0, i32 1
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[dest_a_gep]], i8* align 4 %[[src_a_gep]], i64 1, i1 false)
// CHECK-NEXT: %[[dest_this:.*]] = load %struct.Derived*, %struct.Derived** %[[retval]]
// CHECK-NEXT: ret %struct.Derived* %[[dest_this]]
  bool a : 1;
  bool b : 1;
};

// __declspec(dllexport) causes us to export the implicit copy constructor.
struct __declspec(dllexport) Derived2 : virtual Base {
// CHECK-LABEL: define weak_odr dso_local dllexport x86_thiscallcc noundef %struct.Derived2* @"??0Derived2@@QAE@ABU0@@Z"
// CHECK:      %[[this:.*]] = load %struct.Derived2*, %struct.Derived2** {{.*}}
// CHECK-NEXT: store %struct.Derived2* %[[this]], %struct.Derived2** %[[retval:.*]]
// CHECK:      %[[dest_a_gep:.*]] = getelementptr inbounds %struct.Derived2, %struct.Derived2* %[[this]], i32 0, i32 1
// CHECK-NEXT: %[[src_load:.*]]   = load %struct.Derived2*, %struct.Derived2** {{.*}}
// CHECK-NEXT: %[[src_a_gep:.*]]  = getelementptr inbounds %struct.Derived2, %struct.Derived2* %[[src_load:.*]], i32 0, i32 1
// CHECK-NEXT: %[[dest_a_bitcast:.*]]  = bitcast [1 x i32]* %[[dest_a_gep]] to i8*
// CHECK-NEXT: %[[src_a_bitcast:.*]] = bitcast [1 x i32]* %[[src_a_gep]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %[[dest_a_bitcast]], i8* align 4 %[[src_a_bitcast]], i32 4, i1 false)
// CHECK-NEXT: %[[dest_this:.*]] = load %struct.Derived2*, %struct.Derived2** %[[retval]]
// CHECK-NEXT: ret %struct.Derived2* %[[dest_this]]
  int Array[1];
};
