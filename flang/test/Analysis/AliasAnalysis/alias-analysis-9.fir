// RUN: fir-opt -debug %s -split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' \
// RUN:   2>&1 | FileCheck -match-full-lines %s

// FIXME: Extend much of this to check that it also works after
// convert-hlfir-to-fir, where component access is via fir.coordinate_of instead
// of hlfir.designate.

// FIXME: What about renaming this test to ptr-component.fir?  What about
// merging with alias-analysis-3.fir as it has the same focus?

// module m
// type t
//  type(t), pointer :: next
//  integer :: i
// end type
// contains
// subroutine foo(x, y)
//   type(t) :: x, y
//   integer :: i1, i2
//   i1 = x%next%i
//   x = y
//   i2 = x%next%i
// end subroutine
// end module

// CHECK-LABEL: Testing : "_QMmPfoo"
// x and y are non pointer, non target argument and therefore do not alias.
// CHECK-DAG: x#0 <-> y#0: NoAlias

// y is not a pointer object and therefore does not alias with the x%next
// component.  Also assigning x to y would not modify x.next
// CHECK-DAG: y#0 <-> xnext1#0: NoAlias
// CHECK-DAG: y#0 <-> xnext2#0: NoAlias

// We need to catch the fact that assigning y to x will modify xnext. 
// The only side-effect between the 2 loads of x.next is the assignment to x, 
// therefore x needs to alias with x.next to prevent the loads from being merged.
// CHECK-DAG: x#0 <-> xnext1#0: MayAlias
// CHECK-DAG: x#0 <-> xnext2#0: MayAlias

// TODO: xnext1#0 <-> xnext2#0 are the same and therefore MustAlias but 
//       we are currently not comparing operands involved in offset computations
// CHECK-DAG: xnext1#0 <-> xnext2#0: MayAlias

func.func @_QMmPfoo(%arg0: !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>> {fir.bindc_name = "y"}) {
  %0 = fir.alloca i32 {bindc_name = "i1", uniq_name = "_QMmFfooEi1"}
  %1:2 = hlfir.declare %0 {uniq_name = "_QMmFfooEi1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.alloca i32 {bindc_name = "i2", uniq_name = "_QMmFfooEi2"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QMmFfooEi2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4:2 = hlfir.declare %arg0 {uniq_name = "_QMmFfooEx", test.ptr = "x"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>)
  %5:2 = hlfir.declare %arg1 {uniq_name = "_QMmFfooEy", test.ptr = "y"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>)
  %6 = hlfir.designate %4#0{"next"}   {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "xnext1"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %7 = fir.load %6 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %8 = fir.box_addr %7 : (!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %9 = hlfir.designate %8{"i"} : (!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<i32>
  %10 = fir.load %9 : !fir.ref<i32>
  hlfir.assign %10 to %1#0 : i32, !fir.ref<i32>
  hlfir.assign %5#0 to %4#0 : !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %11 = hlfir.designate %4#0{"next"}   {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "xnext2"} : (!fir.ref<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %12 = fir.load %11 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>>
  %13 = fir.box_addr %12 : (!fir.box<!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>>) -> !fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>
  %14 = hlfir.designate %13{"i"} : (!fir.ptr<!fir.type<_QMmTt{next:!fir.box<!fir.ptr<!fir.type<_QMmTt>>>,i:i32}>>) -> !fir.ref<i32>
  %15 = fir.load %14 : !fir.ref<i32>
  hlfir.assign %15 to %3#0 : i32, !fir.ref<i32>
  return
}

// -----

// The address of a composite aliases the address of any component, including an
// allocatable component.  Like the address of a pointer, the address of an
// allocatable is considered non-data, so AliasAnalysis has special handling to
// detect the aliasing.

// module m
//   type t
//     integer, allocatable :: p
//   end type
//   type(t) :: x
// contains
//   subroutine test()
//     ! access x%p
//   end subroutine
// end module

// CHECK-LABEL: Testing : "_QMmPtest"
// CHECK-DAG: x#0 <-> x%p#0: MayAlias

func.func @_QMmPtest() {
  %0 = fir.address_of(@_QMmEx) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>
  %1:2 = hlfir.declare %0 {test.ptr="x", uniq_name = "_QMmEx"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>)
  %2 = hlfir.designate %1#0{"p"}   {test.ptr="x%p", fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  return
}

// -----

// Nested composites.

// module m
//   type t1
//     integer, pointer :: p
//   end type
//   type t2
//     type(t1) :: x
//     integer, pointer :: p
//     integer :: i
//   end type
// contains
//   subroutine test()
//     type(t2) :: x
//   end subroutine
// end module

// CHECK-LABEL: Testing : "_QMmPtest"

// The addresses of a composite and its pointer component alias even if the
// composite is nested within another composite.
// CHECK-DAG: x#0 <-> x%p#0: MayAlias
// CHECK-DAG: x%x#0 <-> x%x%p#0: MayAlias

// The addresses of different components of the same composite do not alias.
//
// TODO: Thus, all results below should be NoAlias.  However, AliasAnalysis
// normally does not recognize when two values (x%x vs. x%i) are distinct
// components of the same composite (x) as opposed to being potentially
// overlapping parts of something, so it returns MayAlias.  There is special
// handling for a pointer component (x%p) that does recognize it as separate
// from other components (x%i).  But it does not yet distinguish the composite
// (x) from a component (x%x) that is also a composite with a pointer component
// (x%x%p).  Thus, because x and x%p can alias, it assumes x%x and x%p can too.
// CHECK-DAG: x%x#0 <-> x%i#0: MayAlias
// CHECK-DAG: x%p#0 <-> x%i#0: NoAlias
// CHECK-DAG: x%x#0 <-> x%p#0: MayAlias

func.func @_QMmPtest() {
  %0 = fir.alloca !fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}> {bindc_name = "x", uniq_name = "_QMmFtestEx"}
  %1:2 = hlfir.declare %0 {test.ptr="x", uniq_name = "_QMmFtestEx"} : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %2 = hlfir.designate %1#0{"x"}  {test.ptr="x%x"}  : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>>
  %3 = hlfir.designate %1#0{"p"}   {test.ptr="x%p", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %4 = hlfir.designate %1#0{"i"} {test.ptr="x%i"}  : (!fir.ref<!fir.type<_QMmTt2{x:!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>,p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>
  %5 = hlfir.designate %2{"p"}   {test.ptr="x%x%p", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt1{p:!fir.box<!fir.ptr<i32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  return
}

// -----

// Pointer that might dynamically be a component.

// The address of a pointer dummy arg (argp) might alias the address of a
// pointer component (x%p) and thus the address of the associated composite (x),
// but it does not alias the addresses of other components (x%i) of the
// composite.  Moreover, the address *in* argp does not alias any of those.
// Finally, an allocatable dummy arg (arga) should not be mistaken for a pointer
// dummy arg and cannot have such aliasing.

// module m
//   type t
//     integer, pointer :: p
//     integer i
//   end type
//   type(t) :: glob
// contains
//   subroutine test(argp, arga, arg)
//     integer, pointer :: argp
//     integer, allocatable :: arga
//     type(t) :: arg
//     type(t) :: loc
//   end subroutine
// end module

// CHECK-LABEL: Testing : "_QMmPtest"

// Check when composite is a dummy arg.
// CHECK-DAG: argp#0 <-> arg#0: MayAlias
// CHECK-DAG: argp#0 <-> arg%p#0: MayAlias
// CHECK-DAG: argp#0 <-> arg%i#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> arg#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> arg%p#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> arg%i#0: NoAlias
// CHECK-DAG: arga#0 <-> arg#0: NoAlias
// CHECK-DAG: arga#0 <-> arg%p#0: NoAlias

// Check when composite is a global.
// CHECK-DAG: argp#0 <-> glob#0: MayAlias
// CHECK-DAG: argp#0 <-> glob%p#0: MayAlias
// CHECK-DAG: argp#0 <-> glob%i#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> glob#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> glob%p#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> glob%i#0: NoAlias
// CHECK-DAG: arga#0 <-> glob#0: NoAlias
// CHECK-DAG: arga#0 <-> glob%p#0: NoAlias

// Check when composite is a local and thus cannot alias a dummy arg.
//
// TODO: The argp vs. loc%p case should be NoAlias.  However, AliasAnalysis
// currently indiscriminately treats all pointers as aliasing.  That makes sense
// for the addresses within the pointers but not necessarily for the addresses
// of the pointers here.
//
// CHECK-DAG: argp#0 <-> loc#0: NoAlias
// CHECK-DAG: argp#0 <-> loc%p#0: MayAlias
// CHECK-DAG: argp#0 <-> loc%i#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> loc#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> loc%p#0: NoAlias
// CHECK-DAG: argp.tgt#0 <-> loc%i#0: NoAlias
// CHECK-DAG: arga#0 <-> loc#0: NoAlias
// CHECK-DAG: arga#0 <-> loc%p#0: NoAlias

fir.global @_QMmEglob : !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>

func.func @_QMmPtest(%arg0: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "argp"}, %arg1: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "arga"}, %arg2: !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>> {fir.bindc_name = "arg"}) {
  %0 = fir.dummy_scope : !fir.dscope

  %1:2 = hlfir.declare %arg0 dummy_scope %0 {test.ptr="argp", fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMmFtestEargp"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
  %2 = fir.load %1#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %3 = fir.box_addr %2 {test.ptr="argp.tgt"} : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>

  %4:2 = hlfir.declare %arg1 dummy_scope %0 {test.ptr="arga", fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMmFtestEarga"} : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)

  %5:2 = hlfir.declare %arg2 dummy_scope %0 {test.ptr="arg", uniq_name = "_QMmFtestEarg"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %6 = hlfir.designate %5#0{"p"}   {test.ptr="arg%p", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %7 = hlfir.designate %5#0{"i"} {test.ptr="arg%i"}  : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  %8 = fir.address_of(@_QMmEglob) : !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>
  %9:2 = hlfir.declare %8 {test.ptr="glob", uniq_name = "_QMmEglob"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %10 = hlfir.designate %9#0{"p"}   {test.ptr="glob%p", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %11 = hlfir.designate %9#0{"i"} {test.ptr="glob%i"}  : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  %12 = fir.alloca !fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}> {bindc_name = "loc", uniq_name = "_QMmFtestEloc"}
  %13:2 = hlfir.declare %12 {test.ptr="loc", uniq_name = "_QMmFtestEloc"} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>, !fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>)
  %14 = hlfir.designate %13#0{"p"}   {test.ptr="loc%p", fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %15 = hlfir.designate %13#0{"i"} {test.ptr="loc%i"}  : (!fir.ref<!fir.type<_QMmTt{p:!fir.box<!fir.ptr<i32>>,i:i32}>>) -> !fir.ref<i32>

  return
}
