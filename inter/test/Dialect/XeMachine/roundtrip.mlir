// Round-trip: parse, print, re-parse, check.
// RUN: inter-opt %s | inter-opt | FileCheck %s

// CHECK-LABEL: func.func @kernel
// CHECK-SAME: xemachine.kernel_args = [#xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_write", size = 8, alignment = 8, offset = 24>, #xemachine.kernel_arg<kind = by_value, address_space = "none", access = "none", size = 4, alignment = 4, offset = 32>]
// CHECK-SAME: #xemachine.target<chip = "bmg">
func.func @kernel(%arg0: !xemachine.reg<32, -1>, %flag: !xemachine.arf<f, 2, -1>)
    attributes {
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_write", size = 8, alignment = 8, offset = 24>,
        #xemachine.kernel_arg<kind = by_value, address_space = "none", access = "none", size = 4, alignment = 4, offset = 32>
      ],
      xemachine.target = #xemachine.target<chip = "bmg">
    } {
  // CHECK: %[[IMM:.*]] = xemachine.imm 42 : i32
  %imm = xemachine.imm 42 : i32
  // CHECK: %[[ADD:.*]] = xemachine.add %arg0, %[[IMM]] {execSize = 32 : i32} : (!xemachine.reg<32, -1>, !xemachine.imm, i32) -> !xemachine.reg<32, -1>
  %add = xemachine.add %arg0, %imm {execSize = 32 : i32} : (!xemachine.reg<32, -1>, !xemachine.imm, i32) -> !xemachine.reg<32, -1>
  // CHECK: %[[MV:.*]] = xemachine.mov %[[ADD]] : (!xemachine.reg<32, -1>, i32) -> !xemachine.reg<32, -1>
  %mv = xemachine.mov %add : (!xemachine.reg<32, -1>, i32) -> !xemachine.reg<32, -1>
  // CHECK: %[[CSEL:.*]], %[[CSEL_FLAG:.*]] = xemachine.csel eq %[[MV]], %[[MV]], %[[MV]] {execSize = 4 : i32, noMask, signedInt, src0Region = #xemachine.region<4, 4, 1>, src1Region = #xemachine.region<4, 4, 1>, src2Region = #xemachine.region<4, 4, 1>} : (!xemachine.reg<32, -1>, !xemachine.reg<32, -1>, !xemachine.reg<32, -1>, i16) -> (!xemachine.reg<32, -1>, !xemachine.arf<f, 2, 0>)
  %csel, %csel_flag = xemachine.csel eq %mv, %mv, %mv {
      execSize = 4 : i32, noMask, signedInt,
      src0Region = #xemachine.region<4, 4, 1>,
      src1Region = #xemachine.region<4, 4, 1>,
      src2Region = #xemachine.region<4, 4, 1>}
      : (!xemachine.reg<32, -1>, !xemachine.reg<32, -1>,
         !xemachine.reg<32, -1>, i16)
      -> (!xemachine.reg<32, -1>, !xemachine.arf<f, 2, 0>)

  // CHECK: xemachine.send ugm %[[MV]] dep %{{.*}} {desc = 2 : i32, exdesc = 3 : i32, sfid = 1 : i32} : (!xemachine.reg<32, -1>) -> (!xemachine.reg<32, -1>, !xemachine.mem.token)
  %t0 = xemachine.token
  %dst, %t1 = xemachine.send ugm %mv dep %t0 {desc = 2 : i32, exdesc = 3 : i32, sfid = 1 : i32} : (!xemachine.reg<32, -1>) -> (!xemachine.reg<32, -1>, !xemachine.mem.token)
  %t2 = xemachine.after %t1 : !xemachine.mem.token
  %t3 = xemachine.token_join %t0, %t2 : !xemachine.mem.token, !xemachine.mem.token

  // CHECK: xemachine.payload_prologue {
  // CHECK-NEXT: xemachine.payload_prologue_end
  // CHECK-NEXT: }
  xemachine.payload_prologue {
    xemachine.payload_prologue_end
  }

  // CHECK: xemachine.exec_if %{{.*}} : !xemachine.arf<f, 2, -1>
  xemachine.exec_if %flag : !xemachine.arf<f, 2, -1> {
    xemachine.yield
  } otherwise {
    xemachine.yield
  }

  // CHECK: xemachine.uniform_loop
  %loop = xemachine.uniform_loop (%arg0) {
  ^bb0(%cur: !xemachine.reg<32, -1>):
    xemachine.continue_if %flag : !xemachine.arf<f, 2, -1> (%cur : !xemachine.reg<32, -1>)
  } : (!xemachine.reg<32, -1>) -> !xemachine.reg<32, -1>

  return
}
