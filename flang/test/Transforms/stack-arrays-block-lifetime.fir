// RUN: fir-opt %s --stack-arrays | FileCheck %s --check-prefixes=CHECK,SA
// RUN: fir-opt %s --allocation-placement | FileCheck %s --check-prefixes=CHECK,AP
// RUN: fir-opt %s --allocation-placement="legacy=true" | FileCheck %s --check-prefixes=CHECK,AP

func.func private @extent() -> index
func.func private @use(!fir.heap<!fir.char<1,?>>)
func.func private @use_ref(!fir.ref<!fir.char<1,?>>)

// An existing alloca before the temporary must not be reclaimed at block exit.
// CHECK-LABEL: func.func @existing_alloca
// CHECK-NOT: llvm.intr.stacksave
// CHECK: fir.alloca
// CHECK: fir.allocmem
// CHECK: fir.freemem
// CHECK-NOT: llvm.intr.stackrestore
// CHECK: return
func.func @existing_alloca(%again: i1) {
  cf.br ^loop
^loop:
  %n = fir.call @extent() : () -> index
  %a = fir.alloca !fir.char<1,?>(%n : index)
  %s = fir.allocmem !fir.char<1,?>(%n : index)
  fir.call @use(%s) : (!fir.heap<!fir.char<1,?>>) -> ()
  fir.freemem %s : !fir.heap<!fir.char<1,?>>
  cf.cond_br %again, ^loop, ^exit
^exit:
  fir.call @use_ref(%a) : (!fir.ref<!fir.char<1,?>>) -> ()
  return
}

// An allocation whose free is in another block cannot join this block scope.
// CHECK-LABEL: func.func @free_in_successor
// CHECK-NOT: llvm.intr.stacksave
// CHECK: fir.allocmem
// CHECK: fir.freemem
// CHECK-NOT: llvm.intr.stackrestore
// CHECK: return
func.func @free_in_successor(%again: i1) {
  cf.br ^loop
^loop:
  %n = fir.call @extent() : () -> index
  %s = fir.allocmem !fir.char<1,?>(%n : index)
  fir.call @use(%s) : (!fir.heap<!fir.char<1,?>>) -> ()
  cf.br ^cleanup
^cleanup:
  fir.freemem %s : !fir.heap<!fir.char<1,?>>
  cf.cond_br %again, ^loop, ^exit
^exit:
  return
}

// A stack scope properly nested in the block, like the ones lowering emits
// around calls with a runtime-length result, and the stack storage it reclaims
// do not prevent the promotion of a temporary freed inside that scope, which
// reclaims it.
// CHECK-LABEL: func.func @existing_stack_scope
// CHECK: %[[CALL:.*]] = llvm.intr.stacksave
// CHECK: fir.alloca !fir.char<1,?>
// CHECK: fir.alloca !fir.char<1,?>
// CHECK-NOT: fir.allocmem
// CHECK: llvm.intr.stackrestore %[[CALL]]
// CHECK-NOT: llvm.intr.stack
// CHECK: cf.cond_br
func.func @existing_stack_scope(%again: i1) {
  cf.br ^loop
^loop:
  %n = fir.call @extent() : () -> index
  %sp = llvm.intr.stacksave : !llvm.ptr
  %r = fir.alloca !fir.char<1,?>(%n : index)
  fir.call @use_ref(%r) : (!fir.ref<!fir.char<1,?>>) -> ()
  %s = fir.allocmem !fir.char<1,?>(%n : index)
  fir.call @use(%s) : (!fir.heap<!fir.char<1,?>>) -> ()
  fir.freemem %s : !fir.heap<!fir.char<1,?>>
  llvm.intr.stackrestore %sp : !llvm.ptr
  cf.cond_br %again, ^loop, ^exit
^exit:
  return
}

// A temporary allocated inside an existing stack scope but freed after it
// would be reclaimed by that scope while still in use.
// CHECK-LABEL: func.func @freed_after_stack_scope
// CHECK-NOT: fir.alloca
// CHECK: llvm.intr.stacksave
// CHECK-NEXT: fir.allocmem
// CHECK-NEXT: llvm.intr.stackrestore
// CHECK: fir.freemem
func.func @freed_after_stack_scope(%again: i1) {
  cf.br ^loop
^loop:
  %n = fir.call @extent() : () -> index
  %sp = llvm.intr.stacksave : !llvm.ptr
  %s = fir.allocmem !fir.char<1,?>(%n : index)
  llvm.intr.stackrestore %sp : !llvm.ptr
  fir.call @use(%s) : (!fir.heap<!fir.char<1,?>>) -> ()
  fir.freemem %s : !fir.heap<!fir.char<1,?>>
  cf.cond_br %again, ^loop, ^exit
^exit:
  return
}

// The stack scopes of nested blocks reclaim their own stack storage.
// CHECK-LABEL: func.func @nested_stack_scope
// CHECK: %[[BLOCK:.*]] = llvm.intr.stacksave
// CHECK: fir.alloca !fir.char<1,?>
// CHECK: fir.if
// CHECK: %[[INNER:.*]] = llvm.intr.stacksave
// CHECK: fir.alloca !fir.char<1,?>
// CHECK: llvm.intr.stackrestore %[[INNER]]
// CHECK: }
// CHECK-NOT: fir.allocmem
// CHECK: llvm.intr.stackrestore %[[BLOCK]]
// CHECK: cf.cond_br
func.func @nested_stack_scope(%again: i1, %condition: i1) {
  cf.br ^loop
^loop:
  %n = fir.call @extent() : () -> index
  %s = fir.allocmem !fir.char<1,?>(%n : index)
  fir.if %condition {
    %sp = llvm.intr.stacksave : !llvm.ptr
    %r = fir.alloca !fir.char<1,?>(%n : index)
    fir.call @use_ref(%r) : (!fir.ref<!fir.char<1,?>>) -> ()
    llvm.intr.stackrestore %sp : !llvm.ptr
  }
  fir.call @use(%s) : (!fir.heap<!fir.char<1,?>>) -> ()
  fir.freemem %s : !fir.heap<!fir.char<1,?>>
  cf.cond_br %again, ^loop, ^exit
^exit:
  return
}

// A nested alloca outside of a stack scope must not be reclaimed at block exit.
// CHECK-LABEL: func.func @nested_alloca
// CHECK-NOT: llvm.intr.stacksave
// CHECK: fir.allocmem
// CHECK: fir.if
// CHECK: fir.alloca
// CHECK: fir.freemem
// CHECK-NOT: llvm.intr.stackrestore
// CHECK: return
func.func @nested_alloca(%again: i1, %condition: i1) {
  cf.br ^loop
^loop:
  %n = fir.call @extent() : () -> index
  %s = fir.allocmem !fir.char<1,?>(%n : index)
  fir.if %condition {
    %r = fir.alloca !fir.char<1,?>(%n : index)
    fir.call @use_ref(%r) : (!fir.ref<!fir.char<1,?>>) -> ()
  }
  fir.call @use(%s) : (!fir.heap<!fir.char<1,?>>) -> ()
  fir.freemem %s : !fir.heap<!fir.char<1,?>>
  cf.cond_br %again, ^loop, ^exit
^exit:
  return
}

// Allocation placement keeps the character stack preference in device regions,
// and the allocation stays in the region. The stack-arrays pass leaves the
// allocations of offload regions alone.
// CHECK-LABEL: func.func @offload
// CHECK: acc.parallel
// SA-NOT: fir.alloca
// SA: fir.allocmem !fir.char
// SA: fir.freemem
// AP-NOT: fir.allocmem
// AP: fir.alloca !fir.char
// AP-NOT: fir.freemem
// CHECK: acc.yield
func.func @offload(%n: index) {
  acc.parallel {
    %s = fir.allocmem !fir.char<1,?>(%n : index)
    fir.call @use(%s) : (!fir.heap<!fir.char<1,?>>) -> ()
    fir.freemem %s : !fir.heap<!fir.char<1,?>>
    acc.yield
  }
  return
}

// Each block can reclaim its own promoted allocations. A nested allocmem
// does not prevent promotion of the outer block's temporary.
// CHECK-LABEL: func.func @nested_allocations
// CHECK: %[[OUTER:.*]] = llvm.intr.stacksave
// CHECK: fir.alloca !fir.char
// CHECK: fir.if
// CHECK: %[[INNER:.*]] = llvm.intr.stacksave
// CHECK: fir.alloca !fir.char
// CHECK: fir.call @use
// CHECK: llvm.intr.stackrestore %[[INNER]]
// CHECK: }
// CHECK: fir.call @use
// CHECK: llvm.intr.stackrestore %[[OUTER]]
// CHECK: cf.cond_br
// CHECK: return
func.func @nested_allocations(%again: i1, %condition: i1) {
  cf.br ^loop
^loop:
  %n = fir.call @extent() : () -> index
  %outer = fir.allocmem !fir.char<1,?>(%n : index)
  fir.if %condition {
    %m = fir.call @extent() : () -> index
    %inner = fir.allocmem !fir.char<1,?>(%m : index)
    fir.call @use(%inner) : (!fir.heap<!fir.char<1,?>>) -> ()
    fir.freemem %inner : !fir.heap<!fir.char<1,?>>
  }
  fir.call @use(%outer) : (!fir.heap<!fir.char<1,?>>) -> ()
  fir.freemem %outer : !fir.heap<!fir.char<1,?>>
  cf.cond_br %again, ^loop, ^exit
^exit:
  return
}
