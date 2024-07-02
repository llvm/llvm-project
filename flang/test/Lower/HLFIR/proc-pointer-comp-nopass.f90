! Test lowering of NOPASS procedure pointers components.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

module proc_comp_defs
  interface
    real function iface(x)
      real :: x
    end function
    subroutine takes_proc_pointer(p)
      import iface
      procedure(iface), pointer :: p
    end subroutine
  end interface
  type t
    integer :: j
    procedure(iface), nopass, pointer :: p
  end type
end module

real function test1(x)
  use proc_comp_defs, only : t
  type(t) :: x
  test1 = x%p(42.)
end function
! CHECK-LABEL:   func.func @_QPtest1(
! CHECK:           %[[VAL_1:.*]] = fir.alloca f32 {bindc_name = "test1", uniq_name = "_QFtest1Etest1"}
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]]  {{.*}}Etest1
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:           %[[VAL_4:.*]] = arith.constant 4.200000e+01 : f32
! CHECK:           %[[VAL_5:.*]]:3 = hlfir.associate %[[VAL_4]] {adapt.valuebyref} : (f32) -> (!fir.ref<f32>, !fir.ref<f32>, i1)
! CHECK:           %[[VAL_6:.*]] = hlfir.designate %[[VAL_3]]#1{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMproc_comp_defsTt{j:i32,p:!fir.boxproc<(!fir.ref<f32>) -> f32>}>>) -> !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK:           %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.boxproc<(!fir.ref<f32>) -> f32>) -> ((!fir.ref<f32>) -> f32)
! CHECK:           %[[VAL_9:.*]] = fir.call %[[VAL_8]](%[[VAL_5]]#1) fastmath<contract> : (!fir.ref<f32>) -> f32
! CHECK:           hlfir.end_associate %[[VAL_5]]#1, %[[VAL_5]]#2 : !fir.ref<f32>, i1
! CHECK:           hlfir.assign %[[VAL_9]] to %[[VAL_2]]#0 : f32, !fir.ref<f32>

subroutine test2(x)
  use proc_comp_defs, only : t, iface
  type(t) :: x
  procedure(iface) :: ptarget
  x%p => ptarget
end subroutine
! CHECK-LABEL:   func.func @_QPtest2(
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:           %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#1{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMproc_comp_defsTt{j:i32,p:!fir.boxproc<(!fir.ref<f32>) -> f32>}>>) -> !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK:           %[[VAL_3:.*]] = fir.address_of(@_QPptarget) : (!fir.ref<f32>) -> f32
! CHECK:           %[[VAL_4:.*]] = fir.emboxproc %[[VAL_3]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<() -> ()>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.boxproc<() -> ()>) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK:           fir.store %[[VAL_5]] to %[[VAL_2]] : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>

subroutine test3(x)
  use proc_comp_defs, only : t
  type(t) :: x
  x%p => null()
end subroutine
! CHECK-LABEL:   func.func @_QPtest3(
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:           %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#1{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMproc_comp_defsTt{j:i32,p:!fir.boxproc<(!fir.ref<f32>) -> f32>}>>) -> !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK:           %[[VAL_3:.*]] = fir.zero_bits () -> ()
! CHECK:           %[[VAL_4:.*]] = fir.emboxproc %[[VAL_3]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.boxproc<() -> ()>) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK:           fir.store %[[VAL_5]] to %[[VAL_2]] : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>

subroutine test4(x)
  use proc_comp_defs, only : t
  type(t) :: x
  x%p => x%p
end subroutine
! CHECK-LABEL:   func.func @_QPtest4(
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:           %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#1{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMproc_comp_defsTt{j:i32,p:!fir.boxproc<(!fir.ref<f32>) -> f32>}>>) -> !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK:           %[[VAL_3:.*]] = hlfir.designate %[[VAL_1]]#1{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMproc_comp_defsTt{j:i32,p:!fir.boxproc<(!fir.ref<f32>) -> f32>}>>) -> !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_2]] : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>

subroutine test5(x)
  use proc_comp_defs, only : t, takes_proc_pointer
  type(t) :: x
  call takes_proc_pointer(x%p)
end subroutine
! CHECK-LABEL:   func.func @_QPtest5(
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:           %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#1{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMproc_comp_defsTt{j:i32,p:!fir.boxproc<(!fir.ref<f32>) -> f32>}>>) -> !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>) -> !fir.ref<!fir.boxproc<() -> ()>>
! CHECK:           fir.call @_QPtakes_proc_pointer(%[[VAL_3]]) fastmath<contract> : (!fir.ref<!fir.boxproc<() -> ()>>) -> ()

subroutine test6(x)
  use proc_comp_defs, only : t
  type(t) :: x
  nullify(x%p)
end subroutine
! CHECK-LABEL:   func.func @_QPtest6(
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:           %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#1{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMproc_comp_defsTt{j:i32,p:!fir.boxproc<(!fir.ref<f32>) -> f32>}>>) -> !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK:           %[[VAL_3:.*]] = fir.zero_bits () -> ()
! CHECK:           %[[VAL_4:.*]] = fir.emboxproc %[[VAL_3]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.boxproc<() -> ()>) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK:           fir.store %[[VAL_5]] to %[[VAL_2]] : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>

subroutine test7(x, y)
  use proc_comp_defs, only : t
  type(t) :: x, y
  x = y
end subroutine
! CHECK-LABEL:   func.func @_QPtest7(
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]]  {{.*}}Ey
! CHECK:           hlfir.assign %[[VAL_3]]#0 to %[[VAL_2]]#0 : !fir.ref<!fir.type<_QMproc_comp_defsTt{j:i32,p:!fir.boxproc<(!fir.ref<f32>) -> f32>}>>, !fir.ref<!fir.type<_QMproc_comp_defsTt{j:i32,p:!fir.boxproc<(!fir.ref<f32>) -> f32>}>>

subroutine test8(x, y)
  use proc_comp_defs, only : t
  type(t) :: x(10), y(10)
  x = y
end subroutine
! CHECK-LABEL:   func.func @_QPtest8(
! CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]](%[[VAL_3:[a-z0-9]*]])  {{.*}}Ex
! CHECK:           %[[VAL_5:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]](%[[VAL_6:[a-z0-9]*]])  {{.*}}Ey
! CHECK:           hlfir.assign %[[VAL_7]]#0 to %[[VAL_4]]#0 : !fir.ref<!fir.array<10x!fir.type<_QMproc_comp_defsTt{j:i32,p:!fir.boxproc<(!fir.ref<f32>) -> f32>}>>>, !fir.ref<!fir.array<10x!fir.type<_QMproc_comp_defsTt{j:i32,p:!fir.boxproc<(!fir.ref<f32>) -> f32>}>>>
