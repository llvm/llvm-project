// RUN: not inter-timing-dump %s 2>&1 | FileCheck %s

module {
  func.func @invalid_execution_size() {
    %address = xemachine.archreg 0 : !xemachine.reg<16, 0>
    %dst, %token = xemachine.send ugm %address
        {desc = 0 : i32, exdesc = 0 : i32, execSize = 3 : i32,
         noMask, sfid = 0 : i32}
        : (!xemachine.reg<16, 0>)
        -> (!xemachine.reg<16, -1>, !xemachine.mem.token)
    return
  }
}

// CHECK: error: timing model requires a power-of-two execution size no greater than 32
