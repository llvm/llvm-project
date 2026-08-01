// Without the option, calls to _FortranAioOutputDescriptor in device code are
// not reported and the pass succeeds.
// RUN: fir-opt --split-input-file --cuf-transform-device-func %s | FileCheck %s

// With the option enabled, such calls trigger an error and pass failure.
// RUN: fir-opt --split-input-file --cuf-transform-device-func="check-io-output-descriptor=true" \
// RUN:   --verify-diagnostics %s

func.func private @_FortranAioOutputDescriptor(!fir.ref<i8>, !fir.box<none>) -> i1

func.func @_QPsub_aio_device(%arg0: !fir.ref<i8>, %arg1: !fir.box<none>) attributes {cuf.proc_attr = #cuf.cuda_proc<device>} {
  // expected-error@+1 {{descriptor I/O is not supported in device code}}
  %0 = fir.call @_FortranAioOutputDescriptor(%arg0, %arg1) : (!fir.ref<i8>, !fir.box<none>) -> i1
  return
}

// CHECK-LABEL: gpu.module @cuda_device_mod
// CHECK: gpu.func @_QPsub_aio_device

// -----

func.func private @_FortranAioOutputDescriptor(!fir.ref<i8>, !fir.box<none>) -> i1

func.func @_QPsub_aio_host(%arg0: !fir.ref<i8>, %arg1: !fir.box<none>) {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%iv : index) = (%c1 : index) to (%c1 : index) step (%c1 : index) {
    // expected-error@+1 {{descriptor I/O is not supported in device code}}
    %0 = fir.call @_FortranAioOutputDescriptor(%arg0, %arg1) : (!fir.ref<i8>, !fir.box<none>) -> i1
    "fir.end"() : () -> ()
  }
  return
}

// CHECK-LABEL: func.func @_QPsub_aio_host
