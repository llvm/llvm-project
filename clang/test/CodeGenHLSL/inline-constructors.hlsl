// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -std=hlsl202x -emit-llvm -o - -disable-llvm-passes %s | FileCheck %s --check-prefixes=CHECK,NOINLINE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -std=hlsl202x -emit-llvm -o - -disable-llvm-passes %s | FileCheck %s --check-prefixes=CHECK,NOINLINE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -std=hlsl202x -emit-llvm -o - -O0 %s | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -std=hlsl202x -emit-llvm -o - -O0 %s | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -std=hlsl202x -emit-llvm -o - -O1 %s | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -std=hlsl202x -emit-llvm -o - -O1 %s | FileCheck %s --check-prefixes=CHECK,INLINE

// Tests that implicit constructor calls for user classes will always be inlined.

struct Weed {
  Weed() {Count += 1;}
  [[maybe_unused]] void pull() {Count--;}
  static int weedCount() { return Count; }
private:
  static int Count;

} YardWeeds;

int Weed::Count = 1; // It begins. . .

struct Kitty {
  unsigned burrsInFur;

  Kitty() {
    burrsInFur = 0;
  }

  void wanderInYard(int hours) {
    burrsInFur = hours*Weed::weedCount()/8;
  }

  void lick() {
    if(burrsInFur) {
      burrsInFur--;
      Weed w;
    }
  }

} Nion;

void NionsDay(int hours) {
  static Kitty Nion;
  Nion.wanderInYard(hours);
  while(Nion.burrsInFur) Nion.lick();
}

// CHECK:      define void @main()
// CHECK-NEXT: entry:
// Verify constructor is emitted
// NOINLINE-NEXT: call void @_GLOBAL__sub_I_inline_constructors.hlsl()
// NOINLINE-NEXT: %0 = call i32 @llvm.dx.flattened.thread.id.in.group()
// NOINLINE-NEXT: call void @_Z4mainj(i32 %0)
// Verify inlining leaves only calls to "llvm." intrinsics
// INLINE-NOT:    call {{[^@]*}} @{{[^l][^l][^v][^m][^\.]}}
// CHECK:         ret void
[shader("compute")]
[numthreads(1,1,1)]
void main(unsigned GI : SV_GroupIndex) {
  NionsDay(10);
}


// CHECK:      define void @rainyMain()
// CHECK-NEXT: entry:
// Verify constructor is emitted
// NOINLINE-NEXT:   call void @_GLOBAL__sub_I_inline_constructors.hlsl()
// NOINLINE-NEXT:   call void @_Z9rainyMainv()
// Verify inlining leaves only calls to "llvm." intrinsics
// INLINE-NOT:      call {{[^@]*}} @{{[^l][^l][^v][^m][^\.]}}
// CHECK:           ret void
[shader("compute")]
[numthreads(1,1,1)]
void rainyMain() {
  NionsDay(1);
}

