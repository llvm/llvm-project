// RUN: fir-opt --stack-arrays %s | FileCheck %s
// RUN: fir-opt --allocation-placement="legacy=true stack-arrays=true" %s | FileCheck %s
// RUN: fir-opt --allocation-placement %s | FileCheck %s --check-prefix=DEFAULT

// Test that an allocmem made inside an llvm.stacksave/llvm.stackrestore pair
// and still live after the llvm.stackrestore stays on the heap, while one made
// before the llvm.stacksave can be moved to the stack.

func.func private @extent() -> index
func.func private @init(!fir.heap<!fir.array<?xi8>>)
func.func private @use(!fir.heap<!fir.array<?xi8>>)

func.func @size_inside_scope() {
  %sp = llvm.intr.stacksave : !llvm.ptr
  %n = fir.call @extent() : () -> index
  %mem = fir.allocmem !fir.array<?xi8>, %n
  fir.call @init(%mem) : (!fir.heap<!fir.array<?xi8>>) -> ()
  llvm.intr.stackrestore %sp : !llvm.ptr
  fir.call @use(%mem) : (!fir.heap<!fir.array<?xi8>>) -> ()
  fir.freemem %mem : !fir.heap<!fir.array<?xi8>>
  return
}
// CHECK-LABEL: func.func @size_inside_scope(
// CHECK:         fir.allocmem
// CHECK:         fir.freemem

func.func @arg_size_inside_scope(%n: index) {
  %sp = llvm.intr.stacksave : !llvm.ptr
  %mem = fir.allocmem !fir.array<?xi8>, %n
  fir.call @init(%mem) : (!fir.heap<!fir.array<?xi8>>) -> ()
  llvm.intr.stackrestore %sp : !llvm.ptr
  fir.call @use(%mem) : (!fir.heap<!fir.array<?xi8>>) -> ()
  fir.freemem %mem : !fir.heap<!fir.array<?xi8>>
  return
}
// CHECK-LABEL: func.func @arg_size_inside_scope(
// CHECK:         fir.allocmem
// CHECK:         fir.freemem

func.func @nested_scope_in_loop(%again: i1) {
  cf.br ^loop
^loop:
  %n = fir.call @extent() : () -> index
  %outer = llvm.intr.stacksave : !llvm.ptr
  %inner = llvm.intr.stacksave : !llvm.ptr
  %mem = fir.allocmem !fir.char<1,?>(%n : index)
  llvm.intr.stackrestore %inner : !llvm.ptr
  fir.freemem %mem : !fir.heap<!fir.char<1,?>>
  llvm.intr.stackrestore %outer : !llvm.ptr
  cf.cond_br %again, ^loop, ^exit
^exit:
  return
}
// CHECK-LABEL: func.func @nested_scope_in_loop(
// CHECK:         fir.allocmem
// CHECK:         fir.freemem
// DEFAULT-LABEL: func.func @nested_scope_in_loop(
// DEFAULT:         fir.allocmem
// DEFAULT:         fir.freemem
// DEFAULT-LABEL: func.func @restore_of_unknown_save(

func.func @restore_of_unknown_save(%sp: !llvm.ptr, %n: index) {
  %mem = fir.allocmem !fir.array<?xi8>, %n
  llvm.intr.stackrestore %sp : !llvm.ptr
  fir.call @use(%mem) : (!fir.heap<!fir.array<?xi8>>) -> ()
  fir.freemem %mem : !fir.heap<!fir.array<?xi8>>
  return
}
// CHECK-LABEL: func.func @restore_of_unknown_save(
// CHECK:         fir.allocmem
// CHECK:         fir.freemem

func.func @restore_in_region(%cond: i1) {
  %sp = llvm.intr.stacksave : !llvm.ptr
  %n = fir.call @extent() : () -> index
  fir.if %cond {
    %mem = fir.allocmem !fir.array<?xi8>, %n
    fir.call @init(%mem) : (!fir.heap<!fir.array<?xi8>>) -> ()
    llvm.intr.stackrestore %sp : !llvm.ptr
    fir.call @use(%mem) : (!fir.heap<!fir.array<?xi8>>) -> ()
    fir.freemem %mem : !fir.heap<!fir.array<?xi8>>
  }
  return
}
// CHECK-LABEL: func.func @restore_in_region(
// CHECK:         fir.allocmem
// CHECK:         fir.freemem

func.func @restore_on_two_paths(%cond: i1) {
  %sp = llvm.intr.stacksave : !llvm.ptr
  %n = fir.call @extent() : () -> index
  %mem = fir.allocmem !fir.array<?xi8>, %n
  fir.call @init(%mem) : (!fir.heap<!fir.array<?xi8>>) -> ()
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  llvm.intr.stackrestore %sp : !llvm.ptr
  cf.br ^bb3
^bb2:
  llvm.intr.stackrestore %sp : !llvm.ptr
  cf.br ^bb3
^bb3:
  fir.call @use(%mem) : (!fir.heap<!fir.array<?xi8>>) -> ()
  fir.freemem %mem : !fir.heap<!fir.array<?xi8>>
  return
}
// CHECK-LABEL: func.func @restore_on_two_paths(
// CHECK:         fir.allocmem
// CHECK:         fir.freemem

func.func @freed_inside_scope() {
  %sp = llvm.intr.stacksave : !llvm.ptr
  %n = fir.call @extent() : () -> index
  %mem = fir.allocmem !fir.array<?xi8>, %n
  fir.call @use(%mem) : (!fir.heap<!fir.array<?xi8>>) -> ()
  fir.freemem %mem : !fir.heap<!fir.array<?xi8>>
  llvm.intr.stackrestore %sp : !llvm.ptr
  return
}
// CHECK-LABEL: func.func @freed_inside_scope(
// CHECK:         llvm.intr.stacksave
// CHECK:         fir.alloca !fir.array<?xi8>
// CHECK:         llvm.intr.stackrestore
// CHECK-NOT:     fir.freemem

func.func @allocated_before_scope(%n: index) {
  %mem = fir.allocmem !fir.array<?xi8>, %n
  %sp = llvm.intr.stacksave : !llvm.ptr
  fir.call @init(%mem) : (!fir.heap<!fir.array<?xi8>>) -> ()
  llvm.intr.stackrestore %sp : !llvm.ptr
  fir.call @use(%mem) : (!fir.heap<!fir.array<?xi8>>) -> ()
  fir.freemem %mem : !fir.heap<!fir.array<?xi8>>
  return
}
// CHECK-LABEL: func.func @allocated_before_scope(
// CHECK:         fir.alloca !fir.array<?xi8>
// CHECK:         llvm.intr.stacksave
// CHECK:         llvm.intr.stackrestore
// CHECK-NOT:     fir.freemem

func.func @allocated_before_scopes_in_other_blocks(%n: index, %cond: i1) {
  %mem = fir.allocmem !fir.array<?xi8>, %n
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %sp1 = llvm.intr.stacksave : !llvm.ptr
  fir.call @init(%mem) : (!fir.heap<!fir.array<?xi8>>) -> ()
  llvm.intr.stackrestore %sp1 : !llvm.ptr
  cf.br ^bb2
^bb2:
  fir.if %cond {
    %sp2 = llvm.intr.stacksave : !llvm.ptr
    fir.call @init(%mem) : (!fir.heap<!fir.array<?xi8>>) -> ()
    llvm.intr.stackrestore %sp2 : !llvm.ptr
  }
  fir.call @use(%mem) : (!fir.heap<!fir.array<?xi8>>) -> ()
  fir.freemem %mem : !fir.heap<!fir.array<?xi8>>
  return
}
// CHECK-LABEL: func.func @allocated_before_scopes_in_other_blocks(
// CHECK:         fir.alloca !fir.array<?xi8>
// CHECK-NOT:     fir.freemem
// CHECK:         return
