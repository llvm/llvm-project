// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -finclude-default-header -fexperimental-emit-sgep -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -fexperimental-emit-sgep -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIR

struct O {
  int value = 0;

  int get() {
    return value;
  }
};

[shader("compute")]
[numthreads(1,1,1)]
void foo() {
  O o;

// CHECK:      %o = alloca %struct.O, align 1
// CHECK-DXIL: call noundef i32 @_ZN1O3getEv(ptr noundef nonnull align 1 dereferenceable(4) %o)
// CHECK-SPIR: call spir_func noundef i32 @_ZN1O3getEv(ptr noundef nonnull align 1 dereferenceable(4) %o)
  uint tmp = o.get();
}



// CHECK-DXIL: define linkonce_odr hidden noundef i32 @_ZN1O3getEv(ptr noundef nonnull align 1 dereferenceable(4) %this)
// CHECK-SPIR: define linkonce_odr hidden spir_func noundef i32 @_ZN1O3getEv(ptr noundef nonnull align 1 dereferenceable(4) %this)

// CHECK: %[[#A:]] = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.O) %this1, {{i32|i64}} 0)
// CHECK: %[[#B:]] = load i32, ptr %[[#A]], align 1
// CHECK:            ret i32 %[[#B]]
