// RUN: inter-timing-dump %s | FileCheck %s

module {
  func.func @timing() {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %r2 = xemachine.archreg 2 : !xemachine.reg<32, 2>
    %r4 = xemachine.archreg 4 : !xemachine.reg<64, 4>
    %r8 = xemachine.archreg 8 : !xemachine.reg<128, 8>
    %r40 = xemachine.archreg 40 : !xemachine.reg<8, 40>
    %r41 = xemachine.archreg 41 : !xemachine.reg<16, 41>
    %r42 = xemachine.archreg 42 : !xemachine.reg<32, 42>
    %one = xemachine.imm 1 : i32
    %one64 = xemachine.imm 1 : i64

    %mov8 = xemachine.mov %one {execSize = 8 : i32}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %movf = xemachine.mov %r0 {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, f32) -> !xemachine.reg<16, -1>
    %add16 = xemachine.add %r0, %one {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.imm, i32)
        -> !xemachine.reg<16, -1>
    %add32 = xemachine.add %r2, %r2 {execSize = 32 : i32}
        : (!xemachine.reg<32, 2>, !xemachine.reg<32, 2>, i32)
        -> !xemachine.reg<32, -1>
    %add64 = xemachine.add %r4, %one64 {execSize = 32 : i32}
        : (!xemachine.reg<64, 4>, !xemachine.imm, i64)
        -> !xemachine.reg<64, -1>
    %add_acc = xemachine.add %r0, %one {execSize = 16 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.imm, i32)
        -> !xemachine.arf<acc, 16, 0>
    %logic32 = xemachine.or %r2, %r2 {execSize = 32 : i32}
        : (!xemachine.reg<32, 2>, !xemachine.reg<32, 2>, i32)
        -> !xemachine.reg<32, -1>
    %mul = xemachine.mul %r2, %r2 {execSize = 32 : i32}
        : (!xemachine.reg<32, 2>, !xemachine.reg<32, 2>, i32)
        -> !xemachine.arf<acc, 16, 0>
    %mul_a0 = xemachine.mul %r0, %r0 {execSize = 1 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, i32)
        -> !xemachine.arf<a0, 16, 0>
    %mul_mme = xemachine.mul %r0, %r0 {execSize = 1 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, i32)
        -> !xemachine.arf<mme, 16, 0>
    %mul_sr = xemachine.mul %r0, %r0 {execSize = 1 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>, i32)
        -> !xemachine.arf<sr, 16, 0>
    %cmp = xemachine.cmp gt %r2, %r2 {execSize = 16 : i32}
        : (!xemachine.reg<32, 2>, !xemachine.reg<32, 2>, i32)
        -> !xemachine.arf<f, 2, 0>
    %csel, %csel_flag = xemachine.csel eq %r0, %r0, %r0 {
        execSize = 4 : i32, noMask, signedInt,
        src0Region = #xemachine.region<4, 4, 1>,
        src1Region = #xemachine.region<4, 4, 1>,
        src2Region = #xemachine.region<4, 4, 1>}
        : (!xemachine.reg<16, 0>, !xemachine.reg<16, 0>,
           !xemachine.reg<16, 0>, i16)
        -> (!xemachine.reg<16, -1>, !xemachine.arf<f, 2, 1>)
    %a0 = xemachine.and %r0, %one {dstRegion = #xemachine.dstregion<1>,
        dstSub = 2 : i32, execSize = 1 : i32, noMask,
        src0Region = #xemachine.region<0, 1, 0>}
        : (!xemachine.reg<16, 0>, !xemachine.imm, i32)
        -> !xemachine.arf<a0, 16, 0>
    %dpas1 = xemachine.dpas %r40, %r8, %r41 {
        aPrecision = 0 : i32, bPrecision = 0 : i32, elemType = f32,
        execSize = 8 : i32,
        repeatCount = 1 : i32}
        : (!xemachine.reg<8, 40>, !xemachine.reg<128, 8>,
           !xemachine.reg<16, 41>) -> !xemachine.reg<16, 41>
    %dpas2 = xemachine.dpas %r41, %r8, %r42 {
        aPrecision = 0 : i32, bPrecision = 0 : i32, elemType = f32,
        repeatCount = 2 : i32}
        : (!xemachine.reg<16, 41>, !xemachine.reg<128, 8>,
           !xemachine.reg<32, 42>) -> !xemachine.reg<32, 42>
    %dpas8 = xemachine.dpas %r4, %r8, %r8 {
        aPrecision = 0 : i32, bPrecision = 0 : i32, elemType = f32}
        : (!xemachine.reg<64, 4>, !xemachine.reg<128, 8>,
           !xemachine.reg<128, 8>) -> !xemachine.reg<128, 8>

    %root = xemachine.token
    %loaded, %load_token = xemachine.load_a64 %r4 dep %root
        : !xemachine.reg<64, 4>
        -> (!xemachine.reg<32, -1>, !xemachine.mem.token)
    %store_token = xemachine.store_a64 %r4 data %r2 dep %load_token
        : (!xemachine.reg<64, 4>, !xemachine.reg<32, 2>)
        -> !xemachine.mem.token
    %slm, %slm_token = xemachine.load_slm %r2 dep %store_token
        : !xemachine.reg<32, 2>
        -> (!xemachine.reg<32, -1>, !xemachine.mem.token)
    %slm_store = xemachine.store_slm %r2 data %r2 dep %slm_token
        : (!xemachine.reg<32, 2>, !xemachine.reg<32, 2>)
        -> !xemachine.mem.token
    %block, %block_token = xemachine.load_block_a32 %r0 dep %slm_store
        {words = 16 : i32} : !xemachine.reg<16, 0>
        -> (!xemachine.reg<16, -1>, !xemachine.mem.token)
    %readback, %fence_token = xemachine.fence_slm %r0 dep %block_token
        : !xemachine.reg<16, 0>
        -> (!xemachine.reg<16, -1>, !xemachine.mem.token)
    %awaited = xemachine.fence_await %readback dep %fence_token
        : !xemachine.reg<16, -1> -> !xemachine.mem.token
    %barrier = xemachine.barrier_signal %r0 dep %awaited
        : !xemachine.reg<16, 0> -> !xemachine.mem.token
    %raw_dst, %raw_token = xemachine.send ugm %r0 data %r2 dep %barrier
        {desc = 1107354884 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, 0>, !xemachine.reg<32, 2>)
        -> (!xemachine.reg<0, -1>, !xemachine.mem.token)
    %ugm_fence_dst, %ugm_fence = xemachine.send ugm %r0 dep %raw_token
        {desc = 31 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, 0>)
        -> (!xemachine.reg<0, -1>, !xemachine.mem.token)
    %tgm_fence_dst, %tgm_fence = xemachine.send tgm %r0 dep %ugm_fence
        {desc = 31 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, 0>)
        -> (!xemachine.reg<0, -1>, !xemachine.mem.token)
    %slm_dst, %slm_raw = xemachine.send slm %r0 dep %tgm_fence
        {desc = 0 : i32, exdesc = 0 : i32, execSize = 16 : i32,
         noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, 0>)
        -> (!xemachine.reg<16, -1>, !xemachine.mem.token)
    %wide_src_dst, %wide_src = xemachine.send ugm %r0 dep %slm_raw
        {desc = 136316288 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, 0>)
        -> (!xemachine.reg<16, -1>, !xemachine.mem.token)
    %cached_dst, %cached = xemachine.send ugm %r0 dep %wide_src
        {desc = 1645860096 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, 0>)
        -> (!xemachine.reg<16, -1>, !xemachine.mem.token)
    %typed_cached_dst, %typed_cached = xemachine.send tgm %r0 dep %cached
        {desc = 1645860098 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, 0>)
        -> (!xemachine.reg<16, -1>, !xemachine.mem.token)
    %sync = xemachine.sync allrd dep %typed_cached : !xemachine.mem.token
    xemachine.eot %r0 dep %sync : !xemachine.reg<16, 0>
    return
  }
}

// CHECK: xemachine.archreg class=none pipe=none latency=0 occupancy=0 raw-gap=0 war-gap=0 order-gap=0
// CHECK: xemachine.archreg class=none pipe=none latency=0 occupancy=0 raw-gap=0 war-gap=0 order-gap=0
// CHECK: xemachine.archreg class=none pipe=none latency=0 occupancy=0 raw-gap=0 war-gap=0 order-gap=0
// CHECK: xemachine.archreg class=none pipe=none latency=0 occupancy=0 raw-gap=0 war-gap=0 order-gap=0
// CHECK: xemachine.mov class=move-or-logic pipe=integer latency=10 occupancy=1 raw-gap=10 war-gap=2 order-gap=1
// CHECK: xemachine.mov class=move-or-logic pipe=floating latency=10 occupancy=2 raw-gap=10 war-gap=2 order-gap=2
// CHECK: xemachine.add class=arithmetic pipe=integer latency=11 occupancy=2 raw-gap=11 war-gap=2 order-gap=2
// CHECK: xemachine.add class=arithmetic pipe=integer latency=13 occupancy=4 raw-gap=13 war-gap=4 order-gap=4
// CHECK: xemachine.add class=arithmetic pipe=integer latency=13 occupancy=2 raw-gap=13 war-gap=2 order-gap=2
// CHECK: xemachine.add class=accumulator-arithmetic pipe=integer latency=7 occupancy=2 raw-gap=7 war-gap=2 order-gap=2
// CHECK: xemachine.or class=move-or-logic pipe=integer latency=10 occupancy=4 raw-gap=10 war-gap=4 order-gap=4
// CHECK: xemachine.mul class=accumulator-arithmetic pipe=integer latency=9 occupancy=4 raw-gap=9 war-gap=4 order-gap=4
// CHECK: xemachine.mul class=arf-write pipe=integer latency=16 occupancy=1 raw-gap=16 war-gap=2 order-gap=1
// CHECK: xemachine.mul class=accumulator-arithmetic pipe=integer latency=6 occupancy=1 raw-gap=6 war-gap=2 order-gap=1
// CHECK: xemachine.mul class=arithmetic pipe=integer latency=10 occupancy=1 raw-gap=10 war-gap=2 order-gap=1
// CHECK: xemachine.cmp class=arf-write pipe=integer latency=16 occupancy=2 raw-gap=16 war-gap=2 order-gap=2
// CHECK: xemachine.csel class=arf-write pipe=integer latency=16 occupancy=1 raw-gap=16 war-gap=2 order-gap=1
// CHECK: xemachine.and class=arf-write pipe=integer latency=16 occupancy=1 raw-gap=16 war-gap=2 order-gap=1
// CHECK: xemachine.dpas class=systolic pipe=systolic latency=22 occupancy=2 raw-gap=22 war-gap=2 order-gap=2
// CHECK: xemachine.dpas class=systolic pipe=systolic latency=23 occupancy=2 raw-gap=23 war-gap=2 order-gap=2
// CHECK: xemachine.dpas class=systolic pipe=systolic latency=33 occupancy=2 raw-gap=33 war-gap=2 order-gap=2
// CHECK: xemachine.load_a64 class=send pipe=send latency=200 occupancy=4 send-read=12 raw-gap=200 war-gap=12 order-gap=12
// CHECK: xemachine.store_a64 class=send pipe=send latency=200 occupancy=4 send-read=14 raw-gap=200 war-gap=14 order-gap=14
// CHECK: xemachine.load_slm class=send pipe=send latency=45 occupancy=4 send-read=10 raw-gap=45 war-gap=10 order-gap=10
// CHECK: xemachine.store_slm class=send pipe=send latency=45 occupancy=4 send-read=12 raw-gap=45 war-gap=12 order-gap=12
// CHECK: xemachine.load_block_a32 class=send pipe=send latency=45 occupancy=1 send-read=9 raw-gap=45 war-gap=9 order-gap=9
// CHECK: xemachine.fence_slm class=send pipe=send latency=23 occupancy=1 send-read=9 raw-gap=23 war-gap=9 order-gap=9
// CHECK: xemachine.fence_await class=move-or-logic pipe=integer latency=10 occupancy=1 raw-gap=10 war-gap=2 order-gap=1
// CHECK: xemachine.barrier_signal class=send pipe=send latency=30 occupancy=1 send-read=9 raw-gap=30 war-gap=9 order-gap=9
// CHECK: xemachine.send class=send pipe=send latency=200 occupancy=1 send-read=11 raw-gap=200 war-gap=11 order-gap=11
// CHECK: xemachine.send class=send pipe=send latency=35 occupancy=1 send-read=8 raw-gap=35 war-gap=8 order-gap=8
// CHECK: xemachine.send class=send pipe=send latency=60 occupancy=1 send-read=8 raw-gap=60 war-gap=8 order-gap=8
// CHECK: xemachine.send class=send pipe=send latency=28 occupancy=2 send-read=8 raw-gap=28 war-gap=8 order-gap=8
// CHECK: xemachine.send class=send pipe=send latency=200 occupancy=1 send-read=12 raw-gap=200 war-gap=12 order-gap=12
// CHECK: xemachine.send class=send pipe=send latency=45 occupancy=1 send-read=9 raw-gap=45 war-gap=9 order-gap=9
// CHECK: xemachine.send class=send pipe=send latency=75 occupancy=1 send-read=9 raw-gap=75 war-gap=9 order-gap=9
// CHECK: xemachine.sync class=sync pipe=none latency=10 occupancy=1 raw-gap=10 war-gap=2 order-gap=1
// CHECK: xemachine.eot class=send pipe=send latency=50 occupancy=1 send-read=9 raw-gap=50 war-gap=9 order-gap=9
