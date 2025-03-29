// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -o - -disable-llvm-passes %s | FileCheck %s

// Verify that no per variable _Init_thread instructions are emitted for non-trivial static locals
// These would normally be emitted by the MicrosoftCXXABI, but the DirectX backend should exlude them
// Instead, check for the guardvar oparations that should protect the constructor initialization should
// only take place once.

RWBuffer<int> buf[10];

void InitBuf(RWBuffer<int> buf) {
  for (unsigned int i = 0; i < 100; i++)
    buf[i] = 0;
}

// CHECK-NOT: _Init_thread_epoch
// CHECK: define internal void @_Z4mainv
// CHECK-NEXT: entry:
// CHECK-NEXT: [[Tmp1:%.*]] = alloca %"class.hlsl::RWBuffer"
// CHECK-NEXT: [[Tmp2:%.*]] = load i8, ptr @_ZGVZ4mainvE5mybuf
// CHECK-NEXT: [[Tmp3:%.*]] = icmp eq i8 [[Tmp2]], 0
// CHECK-NEXT: br i1 [[Tmp3]]
// CHECK-NOT: _Init_thread_header
// CHECK: init.check:
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIiEC1Ev
// CHECK-NEXT: store i8 1, ptr @_ZGVZ4mainvE5mybuf
// CHECK-NOT: _Init_thread_footer


[shader("compute")]
[numthreads(1,1,1)]
void main() {
  // A non-trivially constructed static local will get checks to verify that it is generated just once
  static RWBuffer<int> mybuf;
  mybuf = buf[0];
  InitBuf(mybuf);
}

