// RUN: fir-opt %s --allocation-placement | FileCheck %s --implicit-check-not=llvm.intr.stacksave --implicit-check-not=llvm.intr.stackrestore
// RUN: fir-opt %s --stack-arrays | FileCheck %s --implicit-check-not=llvm.intr.stacksave --implicit-check-not=llvm.intr.stackrestore
// RUN: fir-opt %s --allocation-placement="legacy=true" | FileCheck %s --implicit-check-not=llvm.intr.stacksave --implicit-check-not=llvm.intr.stackrestore
// RUN: fir-opt %s --allocation-placement="legacy=true stack-arrays=true" | FileCheck %s --implicit-check-not=llvm.intr.stacksave --implicit-check-not=llvm.intr.stackrestore
// RUN: fir-opt %s --allocation-placement="legacy=true stack-arrays=true dynamic-array-on-heap=true" | FileCheck %s --implicit-check-not=llvm.intr.stacksave --implicit-check-not=llvm.intr.stackrestore
// RUN: fir-opt %s --allocation-placement="legacy=true dynamic-array-on-heap=true" | FileCheck %s --check-prefix=HEAP --implicit-check-not=llvm.intr.stacksave --implicit-check-not=llvm.intr.stackrestore --implicit-check-not=fir.alloca

module {
  func.func private @extent() -> index
  func.func private @use(!fir.heap<!fir.char<1,?>>)
  func.func private @use_array(!fir.heap<!fir.array<?xi32>>)

  // Character temporaries prefer the stack even without -fstack-arrays. When
  // emulating the legacy passes, -fdynamic-heap-array keeps them on the heap.
  // HEAP-LABEL: func.func @character_in_cfg_loop
  // HEAP: fir.allocmem !fir.char<1,?>
  // HEAP: fir.freemem
  // CHECK-LABEL: func.func @character_in_cfg_loop
  // CHECK: %[[SP:.*]] = llvm.intr.stacksave
  // CHECK: %[[TEMP:.*]] = fir.alloca !fir.char<1,?>
  // CHECK: fir.call @use
  // CHECK: llvm.intr.stackrestore %[[SP]]
  // CHECK-NOT: fir.freemem
  // CHECK: cf.cond_br
  func.func @character_in_cfg_loop(%again: i1) {
    cf.br ^loop
  ^loop:
    %n = fir.call @extent() : () -> index
    %s = fir.allocmem !fir.char<1,?>(%n : index)
    fir.call @use(%s) : (!fir.heap<!fir.char<1,?>>) -> ()
    fir.freemem %s : !fir.heap<!fir.char<1,?>>
    cf.cond_br %again, ^loop, ^exit
  ^exit:
    return
  }

  // Crossing lifetimes share one block scope, so both can use stack storage.
  // CHECK-LABEL: func.func @crossing_lifetimes
  // CHECK: %[[SP:.*]] = llvm.intr.stacksave
  // CHECK-NOT: llvm.intr.stacksave
  // CHECK-COUNT-2: fir.alloca !fir.char<1,?>
  // CHECK-NOT: llvm.intr.stackrestore
  // CHECK-COUNT-2: fir.call @use
  // CHECK: llvm.intr.stackrestore %[[SP]]
  // CHECK-NOT: llvm.intr.stackrestore
  // CHECK: cf.cond_br
  func.func @crossing_lifetimes(%again: i1) {
    cf.br ^loop
  ^loop:
    %n = fir.call @extent() : () -> index
    %a = fir.allocmem !fir.char<1,?>(%n : index)
    %b = fir.allocmem !fir.char<1,?>(%n : index)
    fir.call @use(%a) : (!fir.heap<!fir.char<1,?>>) -> ()
    fir.freemem %a : !fir.heap<!fir.char<1,?>>
    fir.call @use(%b) : (!fir.heap<!fir.char<1,?>>) -> ()
    fir.freemem %b : !fir.heap<!fir.char<1,?>>
    cf.cond_br %again, ^loop, ^exit
  ^exit:
    return
  }

  // Nested lifetimes use the same single block scope.
  // CHECK-LABEL: func.func @nested_lifetimes
  // CHECK: %[[SP:.*]] = llvm.intr.stacksave
  // CHECK-NOT: llvm.intr.stacksave
  // CHECK-COUNT-2: fir.alloca !fir.char<1,?>
  // CHECK-NOT: llvm.intr.stackrestore
  // CHECK-COUNT-2: fir.call @use
  // CHECK: llvm.intr.stackrestore %[[SP]]
  // CHECK-NOT: llvm.intr.stackrestore
  // CHECK: cf.cond_br
  func.func @nested_lifetimes(%again: i1) {
    cf.br ^loop
  ^loop:
    %n = fir.call @extent() : () -> index
    %a = fir.allocmem !fir.char<1,?>(%n : index)
    %b = fir.allocmem !fir.char<1,?>(%n : index)
    fir.call @use(%b) : (!fir.heap<!fir.char<1,?>>) -> ()
    fir.freemem %b : !fir.heap<!fir.char<1,?>>
    fir.call @use(%a) : (!fir.heap<!fir.char<1,?>>) -> ()
    fir.freemem %a : !fir.heap<!fir.char<1,?>>
    cf.cond_br %again, ^loop, ^exit
  ^exit:
    return
  }
}
