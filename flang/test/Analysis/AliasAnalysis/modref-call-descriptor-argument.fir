// RUN: fir-opt -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' \
// RUN:  --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s

// A call that is passed the address of a descriptor may access the data that
// the descriptor describes, e.g. the data of an allocatable actual argument
// associated with an allocatable dummy. The intent of the dummy applies to
// that data as well. The data of variables that are not passed, and of other
// components of a derived type whose component is passed, stays NoModRef.

// subroutine local()
//   integer, allocatable :: a, b
//   call bump(a)
//   call peek(a)
//   call reset(a)
// end subroutine
//
// The allocatable dummy of bump has no intent, the one of peek is
// intent(in), and the one of reset is intent(out).

// CHECK-LABEL: Testing : "_QPlocal"
// CHECK-DAG: bump_call -> a.desc#0: ModRef{{$}}
// CHECK-DAG: bump_call -> a.data#0: ModRef{{$}}
// CHECK-DAG: bump_call -> b.data#0: NoModRef{{$}}
// CHECK-DAG: peek_call -> a.desc#0: Ref{{$}}
// CHECK-DAG: peek_call -> a.data#0: Ref{{$}}
// CHECK-DAG: peek_call -> b.data#0: NoModRef{{$}}
// CHECK-DAG: reset_call -> a.desc#0: ModRef{{$}}
// CHECK-DAG: reset_call -> a.data#0: ModRef{{$}}
// CHECK-DAG: reset_call -> b.data#0: NoModRef{{$}}
func.func @_QPlocal() {
  %0 = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "a", uniq_name = "_QFlocalEa"}
  %1 = fir.declare %0 uniq_name("_QFlocalEa") fortran_attrs<allocatable> {test.ptr = "a.desc"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "b", uniq_name = "_QFlocalEb"}
  %3 = fir.declare %2 uniq_name("_QFlocalEb") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  fir.call @_QPbump(%1) {test.ptr = "bump_call"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> ()
  fir.call @_QPpeek(%1) {test.ptr = "peek_call"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> ()
  fir.call @_QPreset(%1) {test.ptr = "reset_call"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> ()
  %4 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %5 = fir.box_addr %4 {test.ptr = "a.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %6 = fir.load %3 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %7 = fir.box_addr %6 {test.ptr = "b.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  return
}

// non_recursive subroutine saved()
//   integer, allocatable, save :: a, b
//   call bump(a)
// end subroutine

// CHECK-LABEL: Testing : "_QPsaved"
// CHECK-DAG: bump_call -> a.data#0: ModRef{{$}}
// CHECK-DAG: bump_call -> b.data#0: NoModRef{{$}}
fir.global internal @_QFsavedEa : !fir.box<!fir.heap<i32>> {
  %0 = fir.zero_bits !fir.heap<i32>
  %1 = fir.embox %0 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.has_value %1 : !fir.box<!fir.heap<i32>>
}
fir.global internal @_QFsavedEb : !fir.box<!fir.heap<i32>> {
  %0 = fir.zero_bits !fir.heap<i32>
  %1 = fir.embox %0 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.has_value %1 : !fir.box<!fir.heap<i32>>
}
func.func @_QPsaved() attributes {fir.proc_attrs = #fir.proc_attrs<non_recursive>} {
  %0 = fir.address_of(@_QFsavedEa) : !fir.ref<!fir.box<!fir.heap<i32>>>
  %1 = fir.declare %0 uniq_name("_QFsavedEa") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.address_of(@_QFsavedEb) : !fir.ref<!fir.box<!fir.heap<i32>>>
  %3 = fir.declare %2 uniq_name("_QFsavedEb") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  fir.call @_QPbump(%1) {test.ptr = "bump_call"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> ()
  %4 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %5 = fir.box_addr %4 {test.ptr = "a.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %6 = fir.load %3 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %7 = fir.box_addr %6 {test.ptr = "b.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  return
}

// subroutine dummy(a, b)
//   integer, allocatable :: a, b
//   call bump(a)
// end subroutine

// CHECK-LABEL: Testing : "_QPdummy"
// CHECK-DAG: bump_call -> a.data#0: ModRef{{$}}
// CHECK-DAG: bump_call -> b.data#0: NoModRef{{$}}
func.func @_QPdummy(%arg0: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "b"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFdummyEa") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QFdummyEb") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  fir.call @_QPbump(%1) {test.ptr = "bump_call"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> ()
  %3 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %4 = fir.box_addr %3 {test.ptr = "a.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %5 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %6 = fir.box_addr %5 {test.ptr = "b.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  return
}

// subroutine component()
//   type t
//     integer, allocatable :: a, b
//   end type
//   type(t) :: x
//   call bump(x%a)
// end subroutine

// CHECK-LABEL: Testing : "_QPcomponent"
// CHECK-DAG: bump_call -> x.a.data#0: ModRef{{$}}
// CHECK-DAG: bump_call -> x.b.desc#0: NoModRef{{$}}
// CHECK-DAG: bump_call -> x.b.data#0: NoModRef{{$}}
func.func @_QPcomponent() {
  %0 = fir.alloca !fir.type<_QFcomponentTt{a:!fir.box<!fir.heap<i32>>,b:!fir.box<!fir.heap<i32>>}> {bindc_name = "x", uniq_name = "_QFcomponentEx"}
  %1 = fir.declare %0 uniq_name("_QFcomponentEx") : (!fir.ref<!fir.type<_QFcomponentTt{a:!fir.box<!fir.heap<i32>>,b:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.type<_QFcomponentTt{a:!fir.box<!fir.heap<i32>>,b:!fir.box<!fir.heap<i32>>}>>
  %2 = fir.coordinate_of %1, a : (!fir.ref<!fir.type<_QFcomponentTt{a:!fir.box<!fir.heap<i32>>,b:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  fir.call @_QPbump(%2) {test.ptr = "bump_call"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> ()
  %3 = fir.coordinate_of %1, a : (!fir.ref<!fir.type<_QFcomponentTt{a:!fir.box<!fir.heap<i32>>,b:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %4 = fir.load %3 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %5 = fir.box_addr %4 {test.ptr = "x.a.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %6 = fir.coordinate_of %1, b {test.ptr = "x.b.desc"} : (!fir.ref<!fir.type<_QFcomponentTt{a:!fir.box<!fir.heap<i32>>,b:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %7 = fir.load %6 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %8 = fir.box_addr %7 {test.ptr = "x.b.data"} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  return
}

// subroutine poly()
//   type t
//     integer :: i
//   end type
//   class(t), allocatable :: p
//   call change(p)
// end subroutine
//
// The dummy of change is class(t), allocatable.

// CHECK-LABEL: Testing : "_QPpoly"
// CHECK-DAG: change_call -> p.data#0: ModRef{{$}}
// CHECK-DAG: change_call -> p.i#0: ModRef{{$}}
func.func @_QPpoly() {
  %0 = fir.alloca !fir.class<!fir.heap<!fir.type<_QFpolyTt{i:i32}>>> {bindc_name = "p", uniq_name = "_QFpolyEp"}
  %1 = fir.declare %0 uniq_name("_QFpolyEp") fortran_attrs<allocatable> : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFpolyTt{i:i32}>>>>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QFpolyTt{i:i32}>>>>
  fir.call @_QPchange(%1) {test.ptr = "change_call"} : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFpolyTt{i:i32}>>>>) -> ()
  %2 = fir.load %1 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QFpolyTt{i:i32}>>>>
  %3 = fir.box_addr %2 {test.ptr = "p.data"} : (!fir.class<!fir.heap<!fir.type<_QFpolyTt{i:i32}>>>) -> !fir.heap<!fir.type<_QFpolyTt{i:i32}>>
  %4 = fir.coordinate_of %3, i {test.ptr = "p.i"} : (!fir.heap<!fir.type<_QFpolyTt{i:i32}>>) -> !fir.ref<i32>
  return
}

// subroutine array()
//   integer, allocatable :: a(:)
//   call grow(a)
//   ... a(1) ...
// end subroutine
//
// The dummy of grow is integer, allocatable :: x(:).

// CHECK-LABEL: Testing : "_QParray"
// CHECK-DAG: grow_call -> a.elem#0: ModRef{{$}}
func.func @_QParray() {
  %c1 = arith.constant 1 : index
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "a", uniq_name = "_QFarrayEa"}
  %1 = fir.declare %0 uniq_name("_QFarrayEa") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  fir.call @_QPgrow(%1) {test.ptr = "grow_call"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> ()
  %2 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %3 = fir.array_coor %2 %c1 {test.ptr = "a.elem"} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
  return
}

// The data of a is referenced through an associate name declared before the
// call.
//
// subroutine assoc()
//   integer, allocatable :: a
//   associate (y => a)
//     call bump(a)
//     ... y ...
//   end associate
// end subroutine

// CHECK-LABEL: Testing : "_QPassoc"
// CHECK-DAG: bump_call -> y#0: ModRef{{$}}
func.func @_QPassoc() {
  %0 = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "a", uniq_name = "_QFassocEa"}
  %1 = fir.declare %0 uniq_name("_QFassocEa") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %3 = fir.box_addr %2 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %4 = fir.declare %3 uniq_name("_QFassocEy") {test.ptr = "y"} : (!fir.heap<i32>) -> !fir.heap<i32>
  fir.call @_QPbump(%1) {test.ptr = "bump_call"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> ()
  return
}

func.func private @_QPbump(!fir.ref<!fir.box<!fir.heap<i32>>>)
func.func private @_QPpeek(!fir.ref<!fir.box<!fir.heap<i32>>> {fir.fortran_attrs = #fir.var_attrs<intent_in>})
func.func private @_QPreset(!fir.ref<!fir.box<!fir.heap<i32>>> {fir.fortran_attrs = #fir.var_attrs<intent_out>})
func.func private @_QPchange(!fir.ref<!fir.class<!fir.heap<!fir.type<_QFpolyTt{i:i32}>>>>)
func.func private @_QPgrow(!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
