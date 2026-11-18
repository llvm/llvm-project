// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Struct fields and arrays hold bools in their memory representation (i32).

struct Inner {
  bool2 v : C;
};

struct Input {
  bool b : A;
  bool3 bv : B;
  bool arr[2] : D;
  Inner inner;
};

struct Output {
  bool b : E;
  bool3 bv : F;
  bool arr[2] : G;
  Inner inner;
};

// CHECK: %struct.Output = type { i32, <3 x i32>, [2 x i32], %struct.Inner }
// CHECK: %struct.Inner = type { <2 x i32> }
// CHECK: %struct.Input = type { i32, <3 x i32>, [2 x i32], %struct.Inner }
// CHECK: %struct.ArrOut = type { [2 x <3 x i32>], [2 x [2 x <2 x i32>]] }
// CHECK: %struct.ArrIn = type { [2 x <3 x i32>], [2 x [2 x <2 x i32>]] }

// CHECK-LABEL: define void @main()
// CHECK-DXIL: %[[A:.*]] = call i32 @llvm.dx.load.input.i32(i32 0, i32 0, i8 0, i32 poison)
// CHECK-DXIL: %[[S0:.*]] = insertvalue %struct.Input poison, i32 %[[A]], 0
// CHECK-DXIL: %[[B:.*]] = call <3 x i32> @llvm.dx.load.input.v3i32(i32 1, i32 0, i8 0, i32 poison)
// CHECK-DXIL: %[[S1:.*]] = insertvalue %struct.Input %[[S0]], <3 x i32> %[[B]], 1
// CHECK-DXIL: %[[D0:.*]] = call i32 @llvm.dx.load.input.i32(i32 2, i32 0, i8 0, i32 poison)
// CHECK-DXIL: %[[ARR0:.*]] = insertvalue [2 x i32] poison, i32 %[[D0]], 0
// CHECK-DXIL: %[[D1:.*]] = call i32 @llvm.dx.load.input.i32(i32 2, i32 1, i8 0, i32 poison)
// CHECK-DXIL: %[[ARR1:.*]] = insertvalue [2 x i32] %[[ARR0]], i32 %[[D1]], 1
// CHECK-DXIL: %[[S2:.*]] = insertvalue %struct.Input %[[S1]], [2 x i32] %[[ARR1]], 2
// CHECK-DXIL: %[[C:.*]] = call <2 x i32> @llvm.dx.load.input.v2i32(i32 3, i32 0, i8 0, i32 poison)
// CHECK-DXIL: %[[IN:.*]] = insertvalue %struct.Inner poison, <2 x i32> %[[C]], 0
// CHECK-DXIL: insertvalue %struct.Input %[[S2]], %struct.Inner %[[IN]], 3

// CHECK-SPIRV: load i32, ptr addrspace(7) @A0
// CHECK-SPIRV: load <3 x i32>, ptr addrspace(7) @B0
// CHECK-SPIRV: load [2 x i32], ptr addrspace(7) @D0
// CHECK-SPIRV: load <2 x i32>, ptr addrspace(7) @C0

// CHECK: %[[OUT:.*]] = load %struct.Output, ptr
// CHECK: %[[OB:.*]] = extractvalue %struct.Output %[[OUT]], 0
// CHECK-DXIL: call void @llvm.dx.store.output.i32(i32 0, i32 0, i8 0, i32 %[[OB]])
// CHECK-SPIRV: store i32 %[[OB]], ptr addrspace(8) @E0
// CHECK: %[[OBV:.*]] = extractvalue %struct.Output %[[OUT]], 1
// CHECK-DXIL: call void @llvm.dx.store.output.v3i32(i32 1, i32 0, i8 0, <3 x i32> %[[OBV]])
// CHECK-SPIRV: store <3 x i32> %[[OBV]], ptr addrspace(8) @F0
// CHECK: %[[OARR:.*]] = extractvalue %struct.Output %[[OUT]], 2
// CHECK-DXIL: %[[OARR0:.*]] = extractvalue [2 x i32] %[[OARR]], 0
// CHECK-DXIL: call void @llvm.dx.store.output.i32(i32 2, i32 0, i8 0, i32 %[[OARR0]])
// CHECK-DXIL: %[[OARR1:.*]] = extractvalue [2 x i32] %[[OARR]], 1
// CHECK-DXIL: call void @llvm.dx.store.output.i32(i32 2, i32 1, i8 0, i32 %[[OARR1]])
// CHECK-SPIRV: store [2 x i32] %[[OARR]], ptr addrspace(8) @G0
// CHECK: %[[OIN:.*]] = extractvalue %struct.Output %[[OUT]], 3
// CHECK: %[[OV:.*]] = extractvalue %struct.Inner %[[OIN]], 0
// CHECK-DXIL: call void @llvm.dx.store.output.v2i32(i32 3, i32 0, i8 0, <2 x i32> %[[OV]])
// CHECK-SPIRV: store <2 x i32> %[[OV]], ptr addrspace(8) @C0.1
[shader("vertex")]
Output main(Input input) {
  Output o;
  o.b = input.b;
  o.bv = input.bv;
  o.arr = input.arr;
  o.inner = input.inner;
  return o;
}

// Array parameters are also passed in memory representation.
// CHECK-LABEL: define void @other()
// CHECK-DXIL: %[[H0:.*]] = call i32 @llvm.dx.load.input.i32(i32 0, i32 0, i8 0, i32 poison)
// CHECK-DXIL: %[[H0V:.*]] = insertvalue [2 x i32] poison, i32 %[[H0]], 0
// CHECK-DXIL: %[[H1:.*]] = call i32 @llvm.dx.load.input.i32(i32 0, i32 1, i8 0, i32 poison)
// CHECK-DXIL: insertvalue [2 x i32] %[[H0V]], i32 %[[H1]], 1
// CHECK-SPIRV: load [2 x i32], ptr addrspace(7) @H0
[shader("vertex")]
float other(bool arr[2] : H) : I {
  return arr[1];
}

// Arrays of bool vectors and matrices strip down to the vector or matrix row
// type in memory representation.
struct ArrIn {
  bool3 va[2] : J;
  bool2x2 ma[2] : K;
};

struct ArrOut {
  bool3 va[2] : L;
  bool2x2 ma[2] : M;
};

// CHECK-LABEL: define void @arrays()
// CHECK-DXIL: %[[J0:.*]] = call <3 x i32> @llvm.dx.load.input.v3i32(i32 0, i32 0, i8 0, i32 poison)
// CHECK-DXIL: %[[VA0:.*]] = insertvalue [2 x <3 x i32>] poison, <3 x i32> %[[J0]], 0
// CHECK-DXIL: %[[J1:.*]] = call <3 x i32> @llvm.dx.load.input.v3i32(i32 0, i32 1, i8 0, i32 poison)
// CHECK-DXIL: %[[VA1:.*]] = insertvalue [2 x <3 x i32>] %[[VA0]], <3 x i32> %[[J1]], 1
// CHECK-DXIL: %[[AS0:.*]] = insertvalue %struct.ArrIn poison, [2 x <3 x i32>] %[[VA1]], 0
// CHECK-DXIL: %[[K0:.*]] = call <2 x i32> @llvm.dx.load.input.v2i32(i32 1, i32 0, i8 0, i32 poison)
// CHECK-DXIL: %[[MA0:.*]] = insertvalue [2 x [2 x <2 x i32>]] poison, <2 x i32> %[[K0]], 0, 0
// CHECK-DXIL: %[[K1:.*]] = call <2 x i32> @llvm.dx.load.input.v2i32(i32 1, i32 1, i8 0, i32 poison)
// CHECK-DXIL: %[[MA1:.*]] = insertvalue [2 x [2 x <2 x i32>]] %[[MA0]], <2 x i32> %[[K1]], 0, 1
// CHECK-DXIL: %[[K2:.*]] = call <2 x i32> @llvm.dx.load.input.v2i32(i32 1, i32 2, i8 0, i32 poison)
// CHECK-DXIL: %[[MA2:.*]] = insertvalue [2 x [2 x <2 x i32>]] %[[MA1]], <2 x i32> %[[K2]], 1, 0
// CHECK-DXIL: %[[K3:.*]] = call <2 x i32> @llvm.dx.load.input.v2i32(i32 1, i32 3, i8 0, i32 poison)
// CHECK-DXIL: %[[MA3:.*]] = insertvalue [2 x [2 x <2 x i32>]] %[[MA2]], <2 x i32> %[[K3]], 1, 1
// CHECK-DXIL: insertvalue %struct.ArrIn %[[AS0]], [2 x [2 x <2 x i32>]] %[[MA3]], 1
// CHECK-SPIRV: load [2 x <3 x i32>], ptr addrspace(7) @J0
// CHECK-SPIRV: load [2 x [2 x <2 x i32>]], ptr addrspace(7) @K0
// CHECK: %[[AOUT:.*]] = load %struct.ArrOut, ptr
// CHECK: %[[OVA:.*]] = extractvalue %struct.ArrOut %[[AOUT]], 0
// CHECK-DXIL: %[[OVA0:.*]] = extractvalue [2 x <3 x i32>] %[[OVA]], 0
// CHECK-DXIL: call void @llvm.dx.store.output.v3i32(i32 0, i32 0, i8 0, <3 x i32> %[[OVA0]])
// CHECK-DXIL: %[[OVA1:.*]] = extractvalue [2 x <3 x i32>] %[[OVA]], 1
// CHECK-DXIL: call void @llvm.dx.store.output.v3i32(i32 0, i32 1, i8 0, <3 x i32> %[[OVA1]])
// CHECK-SPIRV: store [2 x <3 x i32>] %[[OVA]], ptr addrspace(8) @L0
// CHECK: %[[OMA:.*]] = extractvalue %struct.ArrOut %[[AOUT]], 1
// CHECK-DXIL: %[[OMA00:.*]] = extractvalue [2 x [2 x <2 x i32>]] %[[OMA]], 0, 0
// CHECK-DXIL: call void @llvm.dx.store.output.v2i32(i32 1, i32 0, i8 0, <2 x i32> %[[OMA00]])
// CHECK-DXIL: %[[OMA11:.*]] = extractvalue [2 x [2 x <2 x i32>]] %[[OMA]], 1, 1
// CHECK-DXIL: call void @llvm.dx.store.output.v2i32(i32 1, i32 3, i8 0, <2 x i32> %[[OMA11]])
// CHECK-SPIRV: store [2 x [2 x <2 x i32>]] %[[OMA]], ptr addrspace(8) @M0
[shader("vertex")]
ArrOut arrays(ArrIn input) {
  ArrOut o;
  o.va = input.va;
  o.ma = input.ma;
  return o;
}
