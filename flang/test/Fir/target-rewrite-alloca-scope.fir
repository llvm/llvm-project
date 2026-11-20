// RUN: fir-opt --target-rewrite="target=x86_64-unknown-linux-gnu" --cfg-conversion --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

// ABI temporaries must not accumulate on the stack on each loop iteration.
func.func private @direct() -> complex<f32>
func.func private @indirect() -> complex<f128>
func.func private @consume(complex<f32>, complex<f128>, !fir.type<t{x:i64,y:i64,z:i64}>)

// CHECK-LABEL: llvm.func @loop
// CHECK: llvm.alloca
// CHECK: llvm.br
// CHECK-NOT: llvm.alloca
// CHECK: llvm.call @direct
// CHECK-NOT: llvm.alloca
// CHECK: llvm.call @indirect
// CHECK-NOT: llvm.alloca
// CHECK: llvm.call @consume
// CHECK-NOT: llvm.alloca
// CHECK: llvm.return
func.func @loop(%n : index, %v : !fir.type<t{x:i64,y:i64,z:i64}>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  fir.do_loop %i = %c0 to %n step %c1 {
    %a = fir.call @direct() : () -> complex<f32>
    %b = fir.call @indirect() : () -> complex<f128>
    fir.call @consume(%a, %b, %v) : (complex<f32>, complex<f128>, !fir.type<t{x:i64,y:i64,z:i64}>) -> ()
  }
  return
}

// CHECK-LABEL: llvm.func @parallel
// CHECK-NOT: llvm.alloca
// CHECK: omp.parallel
// CHECK: llvm.alloca
// CHECK: llvm.br
// CHECK-NOT: llvm.alloca
// CHECK: llvm.call @direct
// CHECK-NOT: llvm.alloca
// CHECK: llvm.call @indirect
// CHECK-NOT: llvm.alloca
// CHECK: llvm.call @consume
// CHECK-NOT: llvm.alloca
// CHECK: omp.terminator
func.func @parallel(%n : index, %v : !fir.type<t{x:i64,y:i64,z:i64}>) {
  omp.parallel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    fir.do_loop %i = %c0 to %n step %c1 {
      %a = fir.call @direct() : () -> complex<f32>
      %b = fir.call @indirect() : () -> complex<f128>
      fir.call @consume(%a, %b, %v) : (complex<f32>, complex<f128>, !fir.type<t{x:i64,y:i64,z:i64}>) -> ()
    }
    omp.terminator
  }
  return
}

// OpenACC recipes own their allocation block too.
// CHECK-LABEL: acc.private.recipe @recipe
// CHECK: llvm.alloca
// CHECK: llvm.br
// CHECK-NOT: llvm.alloca
// CHECK: llvm.call @direct
// CHECK-NOT: llvm.alloca
// CHECK: llvm.call @save
// CHECK-NOT: llvm.alloca
// CHECK: acc.yield
func.func private @save(complex<f32>, !fir.ref<i32>)
acc.private.recipe @recipe : !fir.ref<i32> init {
^bb0(%original: !fir.ref<i32>):
  %storage = fir.alloca i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  fir.do_loop %i = %c0 to %c10 step %c1 {
    %a = fir.call @direct() : () -> complex<f32>
    fir.call @save(%a, %storage) : (complex<f32>, !fir.ref<i32>) -> ()
  }
  acc.yield %storage : !fir.ref<i32>
}
