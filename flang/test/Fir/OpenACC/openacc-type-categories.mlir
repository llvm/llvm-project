// Use --mlir-disable-threading so that the diagnostic printing is serialized.
// RUN: fir-opt %s -pass-pipeline='builtin.module(test-fir-openacc-interfaces)' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

module {
  fir.global linkonce @test_string constant : !fir.char<1,26> {
    %0 = fir.string_lit "hello_world_test_string\00"(26) : !fir.char<1,26>
    fir.has_value %0 : !fir.char<1,26>
  }

  // Test global constant string with pointer conversion
  func.func @_QPtest_global_string_ptr() {
    %0 = fir.address_of(@test_string) : !fir.ref<!fir.char<1,26>>
    %1 = fir.convert %0 : (!fir.ref<!fir.char<1,26>>) -> !fir.ref<i8>
    %2 = acc.copyin varPtr(%1 : !fir.ref<i8>) -> !fir.ref<i8> {name = "test_string", structured = false}
    acc.enter_data dataOperands(%2 : !fir.ref<i8>)
    return
  }

  // CHECK: Visiting: %{{.*}} = acc.copyin varPtr(%{{.*}} : !fir.ref<i8>) -> !fir.ref<i8> {name = "test_string", structured = false}
  // CHECK: Pointer-like and Mappable: !fir.ref<i8>
  // CHECK: Type category: nonscalar

  // Test array with pointer conversion
  func.func @_QPtest_alloca_array_ptr() {
    %c10 = arith.constant 10 : index
    %0 = fir.alloca !fir.array<10xf32> {bindc_name = "local_array", uniq_name = "_QFtest_alloca_array_ptrElocal_array"}
    %1 = fir.convert %0 : (!fir.ref<!fir.array<10xf32>>) -> !fir.ref<i8>
    %2 = acc.copyin varPtr(%1 : !fir.ref<i8>) -> !fir.ref<i8> {name = "local_array", structured = false}
    acc.enter_data dataOperands(%2 : !fir.ref<i8>)
    return
  }

  // CHECK: Visiting: %{{.*}} = acc.copyin varPtr(%{{.*}} : !fir.ref<i8>) -> !fir.ref<i8> {name = "local_array", structured = false}
  // CHECK: Pointer-like and Mappable: !fir.ref<i8>
  // CHECK: Type category: array
}
