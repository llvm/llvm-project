// RUN: fir-opt %s --split-input-file -o /dev/null --mlir-disable-threading \
// RUN:   -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' \
// RUN:   2>&1 | FileCheck -match-full-lines %s

fir.global @device_global1 {data_attr = #cuf.cuda<device>} : i32
fir.global @device_global2 {data_attr = #cuf.cuda<device>} : i32
fir.global @device_target {data_attr = #cuf.cuda<device>, target} : i32
fir.global @device_pointer {data_attr = #cuf.cuda<device>} : !fir.box<!fir.ptr<i32>>

func.func @test_device_address() {
  %0 = fir.address_of(@device_global1) {test.ptr = "host_ptr1"} : !fir.ref<i32>
  %1 = cuf.device_address @device_global1 {test.ptr = "device_ptr1"} -> !fir.ref<i32>
  %2 = fir.address_of(@device_global2) {test.ptr = "host_ptr2"} : !fir.ref<i32>
  %3 = cuf.device_address @device_global2 {test.ptr = "device_ptr2"} -> !fir.ref<i32>
  %4 = fir.address_of(@device_target) {test.ptr = "host_target"} : !fir.ref<i32>
  %5 = cuf.device_address @device_target {test.ptr = "device_target"} -> !fir.ref<i32>
  %6 = fir.address_of(@device_pointer) {test.ptr = "host_pointer"} : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %7 = cuf.device_address @device_pointer {test.ptr = "device_pointer"} -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  return
}

// CHECK-LABEL: Testing : "test_device_address"
// CHECK: host_ptr1#0 <-> device_ptr1#0: MustAlias
// CHECK: host_ptr1#0 <-> host_ptr2#0: NoAlias
// CHECK: device_ptr1#0 <-> host_ptr2#0: NoAlias
// CHECK: host_ptr1#0 <-> device_ptr2#0: NoAlias
// CHECK: device_ptr1#0 <-> device_ptr2#0: NoAlias
// CHECK: host_ptr2#0 <-> device_ptr2#0: MustAlias
// CHECK: host_ptr1#0 <-> host_target#0: NoAlias
// CHECK: device_ptr1#0 <-> host_target#0: NoAlias
// CHECK: host_ptr2#0 <-> host_target#0: NoAlias
// CHECK: device_ptr2#0 <-> host_target#0: NoAlias
// CHECK: host_ptr1#0 <-> device_target#0: NoAlias
// CHECK: device_ptr1#0 <-> device_target#0: NoAlias
// CHECK: host_ptr2#0 <-> device_target#0: NoAlias
// CHECK: device_ptr2#0 <-> device_target#0: NoAlias
// CHECK: host_target#0 <-> device_target#0: MustAlias
// CHECK: host_ptr1#0 <-> host_pointer#0: NoAlias
// CHECK: device_ptr1#0 <-> host_pointer#0: NoAlias
// CHECK: host_ptr2#0 <-> host_pointer#0: NoAlias
// CHECK: device_ptr2#0 <-> host_pointer#0: NoAlias
// CHECK: host_target#0 <-> host_pointer#0: NoAlias
// CHECK: device_target#0 <-> host_pointer#0: NoAlias
// CHECK: host_ptr1#0 <-> device_pointer#0: NoAlias
// CHECK: device_ptr1#0 <-> device_pointer#0: NoAlias
// CHECK: host_ptr2#0 <-> device_pointer#0: NoAlias
// CHECK: device_ptr2#0 <-> device_pointer#0: NoAlias
// CHECK: host_target#0 <-> device_pointer#0: NoAlias
// CHECK: device_target#0 <-> device_pointer#0: NoAlias
// CHECK: host_pointer#0 <-> device_pointer#0: MustAlias
