// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -emit-llvm -disable-llvm-passes -o - -hlsl-entry main %s -debug-info-kind=standalone -dwarf-version=4 | FileCheck %s

struct Pair {
  int First;
  float Second;

  int getFirst() {
	  return this.First;
  }

  float getSecond() {
    return Second;
  }
};

[numthreads(1, 1, 1)]
void main() {
  Pair Vals = {1, 2.0};
  Vals.First = Vals.getFirst();
  Vals.Second = Vals.getSecond();
}

// This tests reference like `this` in HLSL
  // CHECK:       %call = call noundef i32 @"?getFirst@Pair@@QAAHXZ"(ptr noundef nonnull align 4 dereferenceable(8) %Vals)
  // CHECK-NEXT:  %First = getelementptr inbounds nuw %struct.Pair, ptr %Vals, i32 0, i32 0
  // CHECK-NEXT:  store i32 %call, ptr %First, align 4
  // CHECK-NEXT:  %call1 = call noundef float @"?getSecond@Pair@@QAAMXZ"(ptr noundef nonnull align 4 dereferenceable(8) %Vals)
  // CHECK-NEXT:  %Second = getelementptr inbounds nuw %struct.Pair, ptr %Vals, i32 0, i32 1

// CHECK: [[Pair:![0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Pair"
// CHECK: [[getFirst:![0-9]+]] = distinct !DISubprogram(name: "getFirst"
// CHECK-SAME: scope: [[Pair]]
// CHECK: [[FirstThis:![0-9]+]] = !DILocalVariable(name: "this", arg: 1, scope: [[getFirst]], type: [[thisType:![0-9]+]]
// CHECK: [[thisType]] = !DIDerivedType(tag: DW_TAG_reference_type, baseType: [[Pair]], size: 32)
