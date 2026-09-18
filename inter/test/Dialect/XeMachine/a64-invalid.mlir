// RUN: not inter-opt %s 2>&1 | FileCheck %s

// CHECK: error: 'xemachine.load_a64' op requires two address dwords per execution lane
func.func @short_address_payload() {
  %root = xemachine.token
  %address = xemachine.archreg 4 : !xemachine.reg<32, 4>
  %loaded, %token = xemachine.load_a64 %address dep %root
      : !xemachine.reg<32, 4>
        -> (!xemachine.reg<32, 8>, !xemachine.mem.token)
  return
}
