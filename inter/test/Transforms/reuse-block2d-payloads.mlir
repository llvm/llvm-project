// RUN: inter-opt %s --inter-reuse-block2d-payloads | FileCheck %s

module {
  func.func @reuse_offsets(
      %base: !xemachine.reg<16, -1>,
      %address: !xemachine.reg<1, -1>,
      %x: !xemachine.reg<1, -1>,
      %y: !xemachine.reg<1, -1>) {
    %zero = xemachine.imm 0 : i32
    %sixteen = xemachine.imm 16 : i32
    %eight = xemachine.imm 8 : i32
    %one = xemachine.imm 1 : i32
    %minus_sixteen = xemachine.imm -16 : i32
    %minus_eight = xemachine.imm -8 : i32
    %shape = xemachine.imm 1807 : i32
    %width = xemachine.imm 127 : i32
    %height = xemachine.imm 127 : i32
    %pitch = xemachine.imm 127 : i32
    %zero_payload = xemachine.mov %zero {noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
    %address_field = xemachine.mov %address {execSize = 1 : i32, noMask}
        : (!xemachine.reg<1, -1>, i32) -> !xemachine.reg<1, -1>
    %width_field = xemachine.mov %width {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %height_field = xemachine.mov %height {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %pitch_field = xemachine.mov %pitch {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %shape_field = xemachine.mov %shape {execSize = 1 : i32, noMask}
        : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
    %template = xemachine.update_tuple %zero_payload, %address_field,
        %width_field, %height_field, %pitch_field, %shape_field
        {offsets = [0, 2, 3, 4, 7]}
        : (!xemachine.reg<16, -1>, !xemachine.reg<1, -1>,
           !xemachine.reg<1, -1>, !xemachine.reg<1, -1>,
           !xemachine.reg<1, -1>, !xemachine.reg<1, -1>)
        -> !xemachine.reg<16, -1>
    %reference = xemachine.update_tuple %template, %x, %y
        {offsets = [5, 6]}
        : (!xemachine.reg<16, -1>, !xemachine.reg<1, -1>,
           !xemachine.reg<1, -1>) -> !xemachine.reg<16, -1>
    %x16 = xemachine.add %x, %sixteen
        {dstRegion = #xemachine.dstregion<1>, execSize = 1 : i32, noMask,
         src0Region = #xemachine.region<0, 1, 0>}
        : (!xemachine.reg<1, -1>, !xemachine.imm, i32)
        -> !xemachine.reg<1, -1>
    %y8 = xemachine.add %y, %eight
        {dstRegion = #xemachine.dstregion<1>, execSize = 1 : i32, noMask,
         src0Region = #xemachine.region<0, 1, 0>}
        : (!xemachine.reg<1, -1>, !xemachine.imm, i32)
        -> !xemachine.reg<1, -1>
    %offset = xemachine.update_tuple %template, %x16, %y8
        {offsets = [5, 6]}
        : (!xemachine.reg<16, -1>, !xemachine.reg<1, -1>,
           !xemachine.reg<1, -1>) -> !xemachine.reg<16, -1>
    %x1 = xemachine.add %x, %one
        {dstRegion = #xemachine.dstregion<1>, execSize = 1 : i32, noMask,
         src0Region = #xemachine.region<0, 1, 0>}
        : (!xemachine.reg<1, -1>, !xemachine.imm, i32)
        -> !xemachine.reg<1, -1>
    %unaligned = xemachine.update_tuple %template, %x1, %y
        {offsets = [5, 6]}
        : (!xemachine.reg<16, -1>, !xemachine.reg<1, -1>,
           !xemachine.reg<1, -1>) -> !xemachine.reg<16, -1>
    %x_negative = xemachine.add %x, %minus_sixteen
        {dstRegion = #xemachine.dstregion<1>, execSize = 1 : i32, noMask,
         src0Region = #xemachine.region<0, 1, 0>}
        : (!xemachine.reg<1, -1>, !xemachine.imm, i32)
        -> !xemachine.reg<1, -1>
    %y_negative = xemachine.add %y, %minus_eight
        {dstRegion = #xemachine.dstregion<1>, execSize = 1 : i32, noMask,
         src0Region = #xemachine.region<0, 1, 0>}
        : (!xemachine.reg<1, -1>, !xemachine.imm, i32)
        -> !xemachine.reg<1, -1>
    %negative = xemachine.update_tuple %template, %x_negative, %y_negative
        {offsets = [5, 6]}
        : (!xemachine.reg<16, -1>, !xemachine.reg<1, -1>,
           !xemachine.reg<1, -1>) -> !xemachine.reg<16, -1>
    %dst0, %token0 = xemachine.send ugm %reference
        {desc = 37749251 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, -1>)
        -> (!xemachine.reg<64, -1>, !xemachine.mem.token)
    %dst1, %token1 = xemachine.send ugm %offset dep %token0
        {desc = 37749251 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, -1>)
        -> (!xemachine.reg<64, -1>, !xemachine.mem.token)
    %dst2, %token2 = xemachine.send ugm %unaligned dep %token1
        {desc = 37749251 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, -1>)
        -> (!xemachine.reg<64, -1>, !xemachine.mem.token)
    %dst3, %token3 = xemachine.send ugm %negative dep %token2
        {desc = 37749251 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, -1>)
        -> (!xemachine.reg<64, -1>, !xemachine.mem.token)
    %dst4, %token4 = xemachine.send ugm %offset dep %token3
        {desc = 574620163 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, -1>)
        -> (!xemachine.reg<64, -1>, !xemachine.mem.token)
    return
  }
}

// CHECK-LABEL: func.func @reuse_offsets
// CHECK: %[[PAYLOAD:.*]] = xemachine.update_tuple {{.*}} {offsets = [5, 6]}
// CHECK: %[[OFFSET:.*]] = xemachine.update_tuple {{.*}} {offsets = [5, 6]}
// CHECK: %[[UNALIGNED:.*]] = xemachine.update_tuple {{.*}} {offsets = [5, 6]}
// CHECK: xemachine.send ugm %[[PAYLOAD]] {{.*}}exdesc = 0
// CHECK: xemachine.send ugm %[[PAYLOAD]] {{.*}}exdesc = 33619968
// CHECK: xemachine.send ugm %[[UNALIGNED]] {{.*}}exdesc = 0
// CHECK: xemachine.send ugm %[[PAYLOAD]] {{.*}}exdesc = -29425664
// CHECK: xemachine.send ugm %[[OFFSET]] {{.*}}desc = 574620163 {{.*}}exdesc = 0
