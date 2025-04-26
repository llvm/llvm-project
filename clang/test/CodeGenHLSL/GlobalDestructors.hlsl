// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s --check-prefixes=CS,NOINLINE-SPIRV,CHECK
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s --check-prefixes=CS,NOINLINE-DXIL,CHECK
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s --check-prefixes=LIB,NOINLINE-DXIL,CHECK
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -O0 %s -o - | FileCheck %s --check-prefixes=INLINE,CHECK
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -O0 %s -o - | FileCheck %s --check-prefixes=INLINE,CHECK

// Tests that constructors and destructors are appropriately generated for globals
// and that their calls are inlined when AlwaysInline is run
// but global variables are retained for the library profiles

// Make sure global variable for ctors/dtors exist for lib profile.
// LIB:@llvm.global_ctors
// LIB:@llvm.global_dtors
// Make sure global variable for ctors/dtors removed for compute profile.
// CS-NOT:@llvm.global_ctors
// CS-NOT:@llvm.global_dtors

struct Tail {
  Tail() {
    add(1);
  }

  ~Tail() {
    add(-1);
  }

  void add(int V) {
    static int Count = 0;
    Count += V;
  }
};

struct Pupper {
  static int Count;

  Pupper() {
    Count += 1; // :)
  }

  ~Pupper() {
    Count -= 1; // :(
  }
} GlobalPup;

void Wag() {
  static Tail T;
  T.add(0);
}

int Pupper::Count = 0;

[numthreads(1,1,1)]
[shader("compute")]
void main(unsigned GI : SV_GroupIndex) {
  Wag();
}

// CHECK:      define void @main()
// CHECK-NEXT: entry:
// Verify destructor is emitted
// NOINLINE-DXIL-NEXT:   call void @_GLOBAL__sub_I_GlobalDestructors.hlsl()
// NOINLINE-DXIL-NEXT:   %0 = call i32 @llvm.dx.flattened.thread.id.in.group()
// NOINLINE-DXIL-NEXT:   call void @_Z4mainj(i32 %0)
// NOINLINE-DXIL-NEXT:   call void @_GLOBAL__D_a()
// NOINLINE-DXIL-NEXT:   ret void

// NOINLINE-SPIRV-NEXT:   %0 = call token @llvm.experimental.convergence.entry()
// NOINLINE-SPIRV-NEXT:   call spir_func void @_GLOBAL__sub_I_GlobalDestructors.hlsl() [ "convergencectrl"(token %0) ]
// NOINLINE-SPIRV-NEXT:   %1 = call i32 @llvm.spv.flattened.thread.id.in.group()
// NOINLINE-SPIRV-NEXT:   call spir_func void @_Z4mainj(i32 %1) [ "convergencectrl"(token %0) ]
// NOINLINE-SPIRV-NEXT:   call spir_func void @_GLOBAL__D_a() [ "convergencectrl"(token %0) ]
// NOINLINE-SPIRV-NEXT:   ret void

// Verify inlining leaves only calls to "llvm." intrinsics
// INLINE-NOT:   call {{[^@]*}} @{{[^l][^l][^v][^m][^\.]}}
// INLINE:   ret void

// This is really just a sanity check I needed for myself to verify that
// function scope static variables also get destroyed properly.

// NOINLINE-DXIL:       define internal void @_GLOBAL__D_a() [[IntAttr:\#[0-9]+]]
// NOINLINE-DXIL-NEXT:  entry:
// NOINLINE-DXIL-NEXT:    call void @_ZN4TailD1Ev(ptr @_ZZ3WagvE1T)
// NOINLINE-DXIL-NEXT:    call void @_ZN6PupperD1Ev(ptr @GlobalPup)
// NOINLINE-DXIL-NEXT:    ret void

// NOINLINE-SPIRV:      define internal spir_func void @_GLOBAL__D_a() [[IntAttr:\#[0-9]+]]
// NOINLINE-SPIRV-NEXT: entry:
// NOINLINE-SPIRV-NEXT:   %0 = call token @llvm.experimental.convergence.entry()
// NOINLINE-SPIRV-NEXT:   call spir_func void @_ZN4TailD1Ev(ptr addrspacecast (ptr addrspace(10) @_ZZ3WagvE1T to ptr)) [ "convergencectrl"(token %0) ]
// NOINLINE-SPIRV-NEXT:   call spir_func void @_ZN6PupperD1Ev(ptr @GlobalPup) [ "convergencectrl"(token %0) ]
// NOINLINE-SPIRV-NEXT:   ret void

// NOINLINE: attributes [[IntAttr]] = {{.*}} alwaysinline
