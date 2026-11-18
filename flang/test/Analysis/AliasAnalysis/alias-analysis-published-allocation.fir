// An allocation without a declaration whose address is stored into a single
// descriptor becomes the data of the variable that the descriptor describes.
// Its accesses must alias the accesses through that descriptor.

// RUN: fir-opt %s -split-input-file -o /dev/null --mlir-disable-threading \
// RUN:   --allow-unregistered-dialect \
// RUN:   -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' \
// RUN:   2>&1 | FileCheck %s

// The answers do not depend on the source cache.
// RUN: fir-opt %s -split-input-file -o /dev/null --mlir-disable-threading \
// RUN:   --allow-unregistered-dialect \
// RUN:   -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis{source-cache=true}))' \
// RUN:   2>&1 | FileCheck %s

// subroutine dummy_var(j, n)
//   integer, allocatable :: j
//   integer :: n
//   j = n   ! j is not allocated: the assignment allocates a new buffer
// end subroutine
//
// The new buffer becomes the data of the dummy j. It does not alias the other
// dummy n, nor an unrelated allocation that is never stored into a descriptor.

// CHECK-LABEL: Testing : "_QPdummy_var"
// CHECK-DAG: n#0 <-> new#0: NoAlias
// CHECK-DAG: n#0 <-> j.data#0: NoAlias
// CHECK-DAG: new#0 <-> j.data#0: MayAlias
// CHECK-DAG: n#0 <-> tmp#0: NoAlias
// CHECK-DAG: new#0 <-> tmp#0: NoAlias
// CHECK-DAG: j.data#0 <-> tmp#0: NoAlias

func.func @_QPdummy_var(%arg0: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "j"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFdummy_varEj") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QFdummy_varEn") {test.ptr = "n"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.load %2 : !fir.ref<i32>
  %4 = fir.allocmem i32 {test.ptr = "new"}
  fir.store %3 to %4 : !fir.heap<i32>
  %5 = fir.embox %4 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %5 to %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %6 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %7 = fir.box_addr %6 {test.ptr = "j.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %8 = fir.allocmem i32 {test.ptr = "tmp"}
  fir.store %3 to %8 : !fir.heap<i32>
  fir.freemem %8 : !fir.heap<i32>
  return
}

// -----

// subroutine save_var(n)
//   integer :: n
//   integer, allocatable, save :: j
//   j = n
// end subroutine

// CHECK-LABEL: Testing : "_QPsave_var"
// CHECK-DAG: n#0 <-> new#0: NoAlias
// CHECK-DAG: new#0 <-> j.data#0: MayAlias

fir.global internal @_QFsave_varEj : !fir.box<!fir.heap<i32>> {
  %0 = fir.zero_bits !fir.heap<i32>
  %1 = fir.embox %0 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.has_value %1 : !fir.box<!fir.heap<i32>>
}

func.func @_QPsave_var(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QFsave_varEj) : !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %1 uniq_name("_QFsave_varEj") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %3 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFsave_varEn") {test.ptr = "n"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.load %3 : !fir.ref<i32>
  %5 = fir.allocmem i32 {test.ptr = "new"}
  fir.store %4 to %5 : !fir.heap<i32>
  %6 = fir.embox %5 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %6 to %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %7 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %8 = fir.box_addr %7 {test.ptr = "j.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  return
}

// -----

// subroutine local_var(n)
//   integer :: n, k
//   integer, allocatable :: j
//   j = n
// end subroutine
//
// The address reaches the descriptor through conversions and a rebox.

// CHECK-LABEL: Testing : "_QPlocal_var"
// CHECK-DAG: k#0 <-> new#0: NoAlias
// CHECK-DAG: new#0 <-> j.data#0: MayAlias
// CHECK-DAG: k#0 <-> j.data#0: NoAlias

func.func @_QPlocal_var(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "j", uniq_name = "_QFlocal_varEj"}
  %2 = fir.declare %1 uniq_name("_QFlocal_varEj") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %3 = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFlocal_varEk"}
  %4 = fir.declare %3 uniq_name("_QFlocal_varEk") {test.ptr = "k"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %5 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFlocal_varEn") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %6 = fir.load %5 : !fir.ref<i32>
  %7 = fir.allocmem i32 {test.ptr = "new"}
  fir.store %6 to %7 : !fir.heap<i32>
  %8 = fir.convert %7 : (!fir.heap<i32>) -> !fir.ref<i32>
  %9 = fir.convert %8 : (!fir.ref<i32>) -> !fir.heap<i32>
  %10 = fir.embox %9 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  %11 = fir.rebox %10 : (!fir.box<!fir.heap<i32>>) -> !fir.box<!fir.heap<i32>>
  fir.store %11 to %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %12 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %13 = fir.box_addr %12 {test.ptr = "j.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  return
}

// -----

// The new buffer reaches the descriptor through a fir.if result, as in the
// reallocation of an allocatable on assignment before the CFG conversion.
// The fir.if result is either the old or the new data of j.

// CHECK-LABEL: Testing : "_QPif_join"
// CHECK-DAG: n#0 <-> j.old#0: NoAlias
// CHECK-DAG: n#0 <-> new#0: NoAlias
// CHECK-DAG: n#0 <-> joined#0: NoAlias
// CHECK-DAG: j.old#0 <-> new#0: MayAlias
// CHECK-DAG: j.old#0 <-> joined#0: MayAlias
// CHECK-DAG: new#0 <-> joined#0: MayAlias

func.func @_QPif_join(%arg0: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "j"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c0_i64 = arith.constant 0 : i64
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFif_joinEj") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QFif_joinEn") {test.ptr = "n"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %4 = fir.box_addr %3 {test.ptr = "j.old"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %5 = fir.convert %4 : (!fir.heap<i32>) -> i64
  %6 = arith.cmpi eq, %5, %c0_i64 : i64
  %7 = fir.if %6 -> (!fir.heap<i32>) {
    %10 = fir.allocmem i32 {test.ptr = "new"}
    fir.result %10 : !fir.heap<i32>
  } else {
    fir.result %4 : !fir.heap<i32>
  }
  %8 = fir.convert %7 {test.ptr = "joined"} : (!fir.heap<i32>) -> !fir.heap<i32>
  %9 = fir.embox %7 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %9 to %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  return
}

// -----

// The new buffer reaches the descriptor through a block argument, as after
// the CFG conversion.

// CHECK-LABEL: Testing : "_QPcfg_join"
// CHECK-DAG: n#0 <-> new#0: NoAlias
// CHECK-DAG: new#0 <-> j.data#0: MayAlias

func.func @_QPcfg_join(%arg0: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "j"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}, %arg2: i1) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFcfg_joinEj") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QFcfg_joinEn") {test.ptr = "n"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %4 = fir.box_addr %3 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  cf.cond_br %arg2, ^bb1, ^bb2(%4 : !fir.heap<i32>)
^bb1:
  %5 = fir.allocmem i32 {test.ptr = "new"}
  cf.br ^bb2(%5 : !fir.heap<i32>)
^bb2(%6: !fir.heap<i32>):
  %7 = fir.embox %6 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %7 to %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %8 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %9 = fir.box_addr %8 {test.ptr = "j.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  return
}

// -----

// The new buffer reaches the descriptor through arith.select.

// CHECK-LABEL: Testing : "_QPselect_join"
// CHECK-DAG: n#0 <-> new#0: NoAlias
// CHECK-DAG: new#0 <-> j.data#0: MayAlias

func.func @_QPselect_join(%arg0: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "j"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}, %arg2: i1) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFselect_joinEj") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QFselect_joinEn") {test.ptr = "n"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %4 = fir.box_addr %3 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %5 = fir.allocmem i32 {test.ptr = "new"}
  %6 = arith.select %arg2, %5, %4 : !fir.heap<i32>
  %7 = fir.embox %6 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %7 to %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %8 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %9 = fir.box_addr %8 {test.ptr = "j.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  return
}

// -----

// The address is stored into two descriptors, so the allocation has no single
// owner and may alias anything.

// CHECK-LABEL: Testing : "_QPtwo_descriptors"
// CHECK-DAG: n#0 <-> new#0: MayAlias
// CHECK-DAG: new#0 <-> j.data#0: MayAlias
// CHECK-DAG: new#0 <-> k.data#0: MayAlias

func.func @_QPtwo_descriptors(%arg0: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "j"}, %arg1: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "k"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFtwo_descriptorsEj") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QFtwo_descriptorsEk") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %3 = fir.declare %arg2 dummy_scope %0 arg 3 uniq_name("_QFtwo_descriptorsEn") {test.ptr = "n"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.allocmem i32 {test.ptr = "new"}
  %5 = fir.embox %4 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %5 to %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  fir.store %5 to %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %6 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %7 = fir.box_addr %6 {test.ptr = "j.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %8 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %9 = fir.box_addr %8 {test.ptr = "k.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  return
}

// -----

// The address is stored into memory that is not a descriptor, so the analysis
// loses track of it and the allocation may alias anything.

// CHECK-LABEL: Testing : "_QPraw_slot"
// CHECK-DAG: n#0 <-> new#0: MayAlias

func.func @_QPraw_slot(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFraw_slotEn") {test.ptr = "n"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = fir.alloca !fir.heap<i32>
  %3 = fir.allocmem i32 {test.ptr = "new"}
  fir.store %3 to %2 : !fir.ref<!fir.heap<i32>>
  return
}

// -----

// subroutine declared_target(p)
//   integer, pointer :: p
//   integer, target :: x
//   integer :: y
//   p => x
// end subroutine
//
// A declared allocation keeps its own identity when its address is stored
// into a descriptor: the rules for POINTER and TARGET apply.

// CHECK-LABEL: Testing : "_QPdeclared_target"
// CHECK-DAG: x#0 <-> y#0: NoAlias
// CHECK-DAG: x#0 <-> p.tgt#0: MayAlias
// CHECK-DAG: y#0 <-> p.tgt#0: NoAlias

func.func @_QPdeclared_target(%arg0: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "p"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFdeclared_targetEp") fortran_attrs<pointer> : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %2 = fir.alloca i32 {bindc_name = "x", fir.target, uniq_name = "_QFdeclared_targetEx"}
  %3 = fir.declare %2 uniq_name("_QFdeclared_targetEx") fortran_attrs<target> {test.ptr = "x"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %4 = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFdeclared_targetEy"}
  %5 = fir.declare %4 uniq_name("_QFdeclared_targetEy") {test.ptr = "y"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %6 = fir.convert %3 : (!fir.ref<i32>) -> !fir.ptr<i32>
  %7 = fir.embox %6 : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
  fir.store %7 to %1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %8 = fir.load %1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %9 = fir.box_addr %8 {test.ptr = "p.tgt"} : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
  return
}

// -----

// The new buffer is also the actual argument of an inlined procedure. The
// dummy argument declaration of the inlined procedure does not declare the
// allocation, so the allocation still becomes the data of j.

// CHECK-LABEL: Testing : "_QPinlined_dummy"
// CHECK-DAG: new#0 <-> callee.x#0: MustAlias
// CHECK-DAG: new#0 <-> j.data#0: MayAlias
// CHECK-DAG: callee.x#0 <-> j.data#0: MayAlias

func.func @_QPinlined_dummy(%arg0: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "j"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFinlined_dummyEj") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QFinlined_dummyEn") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.load %2 : !fir.ref<i32>
  %4 = fir.allocmem i32 {test.ptr = "new"}
  %5 = fir.convert %4 : (!fir.heap<i32>) -> !fir.ref<i32>
  %6 = fir.dummy_scope : !fir.dscope
  %7 = fir.declare %5 dummy_scope %6 arg 1 uniq_name("_QFcalleeEx") {test.ptr = "callee.x"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  fir.store %3 to %7 : !fir.ref<i32>
  %8 = fir.embox %4 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %8 to %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %9 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %10 = fir.box_addr %9 {test.ptr = "j.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  return
}

// -----

// subroutine inlined_pointer(j, p, n)
//   integer, allocatable :: j
//   integer, pointer :: p
//   integer :: n
//   j = n
//   call callee(j, p)
// contains
//   subroutine callee(x, q)
//     integer, target :: x
//     integer, pointer :: q
//     q => x
//     q = 0
//   end subroutine
// end subroutine
//
// The inlined procedure also stores the address of the new buffer into the
// descriptor of p, so the buffer is not only the data of j: inside the
// inlined procedure, the target of q is the buffer.

// CHECK-LABEL: Testing : "_QPinlined_pointer"
// CHECK-DAG: new#0 <-> q.tgt#0: MayAlias
// CHECK-DAG: callee.x#0 <-> q.tgt#0: MayAlias

func.func @_QPinlined_pointer(%arg0: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "j"}, %arg1: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "p"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFinlined_pointerEj") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QFinlined_pointerEp") fortran_attrs<pointer> : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %3 = fir.declare %arg2 dummy_scope %0 arg 3 uniq_name("_QFinlined_pointerEn") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.load %3 : !fir.ref<i32>
  %5 = fir.allocmem i32 {test.ptr = "new"}
  fir.store %4 to %5 : !fir.heap<i32>
  %6 = fir.embox %5 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %6 to %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %7 = fir.convert %5 : (!fir.heap<i32>) -> !fir.ref<i32>
  %8 = fir.dummy_scope : !fir.dscope
  %9 = fir.declare %7 dummy_scope %8 arg 1 uniq_name("_QFinlined_pointerFcalleeEx") fortran_attrs<target> {test.ptr = "callee.x"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %10 = fir.declare %2 dummy_scope %8 arg 2 uniq_name("_QFinlined_pointerFcalleeEq") fortran_attrs<pointer> : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %11 = fir.convert %9 : (!fir.ref<i32>) -> !fir.ptr<i32>
  %12 = fir.embox %11 : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
  fir.store %12 to %10 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %13 = fir.load %10 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %14 = fir.box_addr %13 {test.ptr = "q.tgt"} : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
  %c0_i32 = arith.constant 0 : i32
  fir.store %c0_i32 to %14 : !fir.ptr<i32>
  return
}

// -----

// An operation that the analysis does not know uses the address of the new
// buffer. The operation may store the address anywhere, so the buffer may
// alias anything.

// CHECK-LABEL: Testing : "_QPunknown_use"
// CHECK-DAG: n#0 <-> new#0: MayAlias
// CHECK-DAG: new#0 <-> j.data#0: MayAlias

func.func @_QPunknown_use(%arg0: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "j"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFunknown_useEj") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QFunknown_useEn") {test.ptr = "n"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.allocmem i32 {test.ptr = "new"}
  %4 = fir.convert %3 : (!fir.heap<i32>) -> !fir.ref<i32>
  "test.unknown"(%4) : (!fir.ref<i32>) -> ()
  %5 = fir.embox %3 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %5 to %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %6 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %7 = fir.box_addr %6 {test.ptr = "j.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  return
}

// -----

// The address of the new buffer is returned to the caller, which may store
// it anywhere.

// CHECK-LABEL: Testing : "_QPreturned"
// CHECK-DAG: n#0 <-> new#0: MayAlias

func.func @_QPreturned(%arg0: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "j"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) -> !fir.heap<i32> {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFreturnedEj") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QFreturnedEn") {test.ptr = "n"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.allocmem i32 {test.ptr = "new"}
  %4 = fir.embox %3 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %4 to %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  return %3 : !fir.heap<i32>
}

// -----

// The address of the new buffer is stored into memory as a part of an
// aggregate.

// CHECK-LABEL: Testing : "_QPaggregate"
// CHECK-DAG: n#0 <-> new#0: MayAlias

func.func @_QPaggregate(%arg0: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "j"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFaggregateEj") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QFaggregateEn") {test.ptr = "n"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.allocmem i32 {test.ptr = "new"}
  %4 = fir.embox %3 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %4 to %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %5 = fir.alloca tuple<i64, !fir.heap<i32>>
  %6 = fir.undefined tuple<i64, !fir.heap<i32>>
  %7 = fir.insert_value %6, %3, [1 : index] : (tuple<i64, !fir.heap<i32>>, !fir.heap<i32>) -> tuple<i64, !fir.heap<i32>>
  fir.store %7 to %5 : !fir.ref<tuple<i64, !fir.heap<i32>>>
  return
}

// -----

// The new buffer is passed to a procedure and its address is compared. Neither
// stores the address, so the buffer still becomes the data of j.

// CHECK-LABEL: Testing : "_QPcall_and_compare"
// CHECK-DAG: n#0 <-> new#0: NoAlias
// CHECK-DAG: n#0 <-> j.data#0: NoAlias
// CHECK-DAG: new#0 <-> j.data#0: MayAlias

func.func private @_QPsub(!fir.ref<i32>)

func.func @_QPcall_and_compare(%arg0: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "j"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c0_i64 = arith.constant 0 : i64
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFcall_and_compareEj") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QFcall_and_compareEn") {test.ptr = "n"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.allocmem i32 {test.ptr = "new"}
  %4 = fir.convert %3 : (!fir.heap<i32>) -> !fir.ref<i32>
  fir.call @_QPsub(%4) : (!fir.ref<i32>) -> ()
  %5 = fir.convert %3 : (!fir.heap<i32>) -> i64
  %6 = arith.cmpi ne, %5, %c0_i64 : i64
  %7 = fir.embox %3 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %7 to %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %8 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %9 = fir.box_addr %8 {test.ptr = "j.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  return
}

// -----

// An expression has no address. The assignment copies the value of the
// expression into x, so the storage of the expression stays distinct from x.
// The storage gets an address only in the variable that hlfir.associate
// returns, and that address may be stored anywhere by an unknown operation.

// CHECK-LABEL: Testing : "_QPexpr_assign"
// CHECK-DAG: x#0 <-> tmp#0: NoAlias

// CHECK-LABEL: Testing : "_QPexpr_associate"
// CHECK-DAG: x#0 <-> tmp#0: MayAlias

func.func @_QPexpr_assign(%arg0: !fir.ref<!fir.array<2xi32>> {fir.bindc_name = "x"}) {
  %c2 = arith.constant 2 : index
  %true = arith.constant true
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c2 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg0(%1) dummy_scope %0 arg 1 uniq_name("_QFexpr_assignEx") {test.ptr = "x"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
  %3 = fir.allocmem !fir.array<2xi32> {bindc_name = ".tmp.arrayctor", uniq_name = ""}
  %4:2 = hlfir.declare %3(%1) uniq_name(".tmp.arrayctor") : (!fir.heap<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<2xi32>>, !fir.heap<!fir.array<2xi32>>)
  %5 = hlfir.as_expr %4#0 move %true {test.ptr = "tmp"} : (!fir.heap<!fir.array<2xi32>>, i1) -> !hlfir.expr<2xi32>
  hlfir.assign %5 to %2#0 : !hlfir.expr<2xi32>, !fir.ref<!fir.array<2xi32>>
  hlfir.destroy %5 : !hlfir.expr<2xi32>
  return
}

func.func @_QPexpr_associate(%arg0: !fir.ref<!fir.array<2xi32>> {fir.bindc_name = "x"}) {
  %c2 = arith.constant 2 : index
  %true = arith.constant true
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c2 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg0(%1) dummy_scope %0 arg 1 uniq_name("_QFexpr_associateEx") {test.ptr = "x"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
  %3 = fir.allocmem !fir.array<2xi32> {bindc_name = ".tmp.arrayctor", uniq_name = ""}
  %4:2 = hlfir.declare %3(%1) uniq_name(".tmp.arrayctor") : (!fir.heap<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<2xi32>>, !fir.heap<!fir.array<2xi32>>)
  %5 = hlfir.as_expr %4#0 move %true {test.ptr = "tmp"} : (!fir.heap<!fir.array<2xi32>>, i1) -> !hlfir.expr<2xi32>
  %6:3 = hlfir.associate %5(%1) uniq_name("adapt.valuebyref") : (!hlfir.expr<2xi32>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>, i1)
  "test.unknown"(%6#0) : (!fir.ref<!fir.array<2xi32>>) -> ()
  hlfir.end_associate %6#1, %6#2 : !fir.ref<!fir.array<2xi32>>, i1
  hlfir.destroy %5 : !hlfir.expr<2xi32>
  return
}

// -----

// The address of the new buffer is stored into a descriptor that is a
// component of the buffer, so the source of the descriptor depends on the
// source of the buffer. new.v and self.data.v are the same memory.

// CHECK-LABEL: Testing : "_QPcycle"
// CHECK-DAG: new#0 <-> self.data#0: MayAlias
// CHECK-DAG: self.data#0 <-> new.v#0: MayAlias
// CHECK-DAG: self.data.v#0 <-> new.v#0: MayAlias

func.func @_QPcycle(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFcycleEn") {test.ptr = "n"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = fir.allocmem !fir.type<_QFcycleTnode{v:i32,self:!fir.box<!fir.heap<!fir.type<_QFcycleTnode>>>}> {test.ptr = "new"}
  %3 = fir.coordinate_of %2, self {test.ptr = "self"} : (!fir.heap<!fir.type<_QFcycleTnode{v:i32,self:!fir.box<!fir.heap<!fir.type<_QFcycleTnode>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<_QFcycleTnode{v:i32,self:!fir.box<!fir.heap<!fir.type<_QFcycleTnode>>>}>>>>
  %4 = fir.embox %2 : (!fir.heap<!fir.type<_QFcycleTnode{v:i32,self:!fir.box<!fir.heap<!fir.type<_QFcycleTnode>>>}>>) -> !fir.box<!fir.heap<!fir.type<_QFcycleTnode{v:i32,self:!fir.box<!fir.heap<!fir.type<_QFcycleTnode>>>}>>>
  fir.store %4 to %3 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFcycleTnode{v:i32,self:!fir.box<!fir.heap<!fir.type<_QFcycleTnode>>>}>>>>
  %5 = fir.load %3 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFcycleTnode{v:i32,self:!fir.box<!fir.heap<!fir.type<_QFcycleTnode>>>}>>>>
  %6 = fir.box_addr %5 {test.ptr = "self.data"} : (!fir.box<!fir.heap<!fir.type<_QFcycleTnode{v:i32,self:!fir.box<!fir.heap<!fir.type<_QFcycleTnode>>>}>>>) -> !fir.heap<!fir.type<_QFcycleTnode{v:i32,self:!fir.box<!fir.heap<!fir.type<_QFcycleTnode>>>}>>
  %7 = fir.coordinate_of %6, v {test.ptr = "self.data.v"} : (!fir.heap<!fir.type<_QFcycleTnode{v:i32,self:!fir.box<!fir.heap<!fir.type<_QFcycleTnode>>>}>>) -> !fir.ref<i32>
  %8 = fir.coordinate_of %2, v {test.ptr = "new.v"} : (!fir.heap<!fir.type<_QFcycleTnode{v:i32,self:!fir.box<!fir.heap<!fir.type<_QFcycleTnode>>>}>>) -> !fir.ref<i32>
  return
}
