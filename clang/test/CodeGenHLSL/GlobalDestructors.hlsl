// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -std=hlsl202x -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s --check-prefixes=CS,NOINLINE,CHECK
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -std=hlsl202x -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s --check-prefixes=LIB,NOINLINE,CHECK
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -std=hlsl202x -emit-llvm -O0 %s -o - | FileCheck %s --check-prefixes=INLINE,CHECK
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -std=hlsl202x -emit-llvm -O0 %s -o - | FileCheck %s --check-prefixes=INLINE,CHECK

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
// NOINLINE-NEXT:   call void @_GLOBAL__sub_I_GlobalDestructors.hlsl()
// NOINLINE-NEXT:   %0 = call i32 @llvm.dx.flattened.thread.id.in.group()
// NOINLINE-NEXT:   call void @"?main@@YAXI@Z"(i32 %0)
// NOINLINE-NEXT:   call void @_GLOBAL__D_a()
// NOINLINE-NEXT:   ret void
// Verify inlining leaves only calls to "llvm." intrinsics
// INLINE-NOT:   call {{[^@]*}} @{{[^l][^l][^v][^m][^\.]}}
// INLINE:   ret void

// This is really just a sanity check I needed for myself to verify that
// function scope static variables also get destroyed properly.

// NOINLINE: define internal void @_GLOBAL__D_a() [[IntAttr:\#[0-9]+]]
// NOINLINE-NEXT: entry:
// NOINLINE-NEXT:   call void @"??1Tail@@QAA@XZ"(ptr @"?T@?1??Wag@@YAXXZ@4UTail@@A")
// NOINLINE-NEXT:   call void @"??1Pupper@@QAA@XZ"(ptr @"?GlobalPup@@3UPupper@@A")
// NOINLINE-NEXT:   ret void

// NOINLINE: attributes [[IntAttr]] = {{.*}} alwaysinline
