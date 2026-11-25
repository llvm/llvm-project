// RUN: fir-opt %s --stack-arrays | FileCheck %s --implicit-check-not=llvm.intr.stacksave --implicit-check-not=llvm.intr.stackrestore
// RUN: fir-opt %s --allocation-placement="stack-arrays=true" | FileCheck %s --implicit-check-not=llvm.intr.stacksave --implicit-check-not=llvm.intr.stackrestore

module {
  func.func private @extent() -> index
  func.func private @use(!fir.heap<!fir.array<?xi32>>)

  // Both allocations are reclaimed at block exit. Their crossing individual
  // lifetimes do not require separate stack scopes or heap fallback.
  // CHECK-LABEL: func.func @crossing
  // CHECK: %[[SP:.*]] = llvm.intr.stacksave
  // CHECK-NOT: llvm.intr.stacksave
  // CHECK-COUNT-2: fir.alloca !fir.array<?xi32>
  // CHECK-NOT: llvm.intr.stackrestore
  // CHECK-COUNT-2: fir.call @use
  // CHECK: llvm.intr.stackrestore %[[SP]]
  // CHECK-NOT: llvm.intr.stackrestore
  // CHECK: cf.cond_br
  func.func @crossing(%again: i1) {
    cf.br ^loop
  ^loop:
    %n = fir.call @extent() : () -> index
    %a = fir.allocmem !fir.array<?xi32>, %n
    %b = fir.allocmem !fir.array<?xi32>, %n
    fir.call @use(%a) : (!fir.heap<!fir.array<?xi32>>) -> ()
    fir.freemem %a : !fir.heap<!fir.array<?xi32>>
    fir.call @use(%b) : (!fir.heap<!fir.array<?xi32>>) -> ()
    fir.freemem %b : !fir.heap<!fir.array<?xi32>>
    cf.cond_br %again, ^loop, ^exit
  ^exit:
    return
  }
}
