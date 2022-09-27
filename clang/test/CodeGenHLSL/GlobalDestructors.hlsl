// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -std=hlsl202x -S -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -std=hlsl202x -S -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s --check-prefixes=CHECK

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
void main(unsigned GI : SV_GroupIndex) {
  Wag();
}

//CHECK:      define void @main()
//CHECK-NEXT: entry:
//CHECK-NEXT:   call void @_GLOBAL__sub_I_GlobalDestructors.hlsl()
//CHECK-NEXT:   %0 = call i32 @llvm.dx.flattened.thread.id.in.group()
//CHECK-NEXT:   call void @"?main@@YAXI@Z"(i32 %0)
//CHECK-NEXT:   call void @_GLOBAL__D_a()
//CHECK-NEXT:   ret void

// This is really just a sanity check I needed for myself to verify that
// function scope static variables also get destroyed properly.

//CHECK: define internal void @_GLOBAL__D_a()
//CHECK-NEXT: entry:
//CHECK-NEXT:   call void @"??1Tail@@QAA@XZ"(ptr @"?T@?1??Wag@@YAXXZ@4UTail@@A")
//CHECK-NEXT:   call void @"??1Pupper@@QAA@XZ"(ptr @"?GlobalPup@@3UPupper@@A")
//CHECK-NEXT:   ret void
