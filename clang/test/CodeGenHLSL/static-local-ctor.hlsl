// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -o - -disable-llvm-passes %s | \
// RUN:   llvm-cxxfilt | FileCheck %s

// Verify that no per variable _Init_thread instructions are emitted for non-trivial static locals
// These would normally be emitted by the MicrosoftCXXABI, but the DirectX backend should exlude them
// Instead, check for the guardvar operations that should protect the constructor initialization should
// only take place once.

RWBuffer<int> buf[10];

// CHECK: @[[main_mybuf:.*]] = internal global %"class.hlsl::RWBuffer" zeroinitializer, align 4
// CHECK: @[[main_mybuf_guard:.*]] = internal global i8 0, align 1

void InitBuf(RWBuffer<int> buf) {
  for (unsigned int i = 0; i < 100; i++)
    buf[i] = 0;
}

// CHECK-NOT: _Init_thread_epoch
// CHECK: define internal void @main
// CHECK-NEXT: entry:
// CHECK-NEXT: [[Tmp0:%.*]] = alloca %"class.hlsl::RWBuffer"
// CHECK-NEXT: [[Tmp1:%.*]] = alloca %"class.hlsl::RWBuffer"
// CHECK-NEXT: [[Tmp2:%.*]] = load i8, ptr @guard variable for main()::mybuf
// CHECK-NEXT: [[Tmp3:%.*]] = icmp eq i8 [[Tmp2]], 0
// CHECK-NEXT: br i1 [[Tmp3]]
// CHECK-NOT: _Init_thread_header
// CHECK: init.check:
// CHECK-NEXT: call void @hlsl::RWBuffer<int>::__createFromImplicitBinding
// CHECK-NEXT: call void @hlsl::RWBuffer<int>::RWBuffer(hlsl::RWBuffer<int> const&)(ptr {{.*}} @main()::mybuf, ptr {{.*}}) #
// CHECK-NEXT: store i8 1, ptr @guard variable for main()::mybuf
// CHECK-NOT: _Init_thread_footer

[shader("compute")]
[numthreads(1,1,1)]
void main() {
  // A non-trivially constructed static local will get checks to verify that it is generated just once
  static RWBuffer<int> mybuf = buf[0];
  InitBuf(mybuf);
}
