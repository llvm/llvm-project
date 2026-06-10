! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module m
  implicit none

  ! Array globals that should get alignment 64 by default.
  integer :: int_array(10)
  ! CHECK-DAG: fir.global @_QMmEint_array {alignment = 64 : i64} : !fir.array<10xi32>
  real :: real_array(5, 5)
  ! CHECK-DAG: fir.global @_QMmEreal_array {alignment = 64 : i64} : !fir.array<5x5xf32>
  complex :: complex_array(3)
  ! CHECK-DAG: fir.global @_QMmEcomplex_array {alignment = 64 : i64} : !fir.array<3xcomplex<f32>>
  logical :: logical_array(4)
  ! CHECK-DAG: fir.global @_QMmElogical_array {alignment = 64 : i64} : !fir.array<4x!fir.logical<4>>
  character(len=10) :: char_array(2)
  ! CHECK-DAG: fir.global @_QMmEchar_array {alignment = 64 : i64} : !fir.array<2x!fir.char<1,10>>

  integer, target :: target_array(8)
  ! CHECK-DAG: fir.global @_QMmEtarget_array {alignment = 64 : i64} target : !fir.array<8xi32>

  ! Currently not align 64
  integer, allocatable :: alloc_array(:)
  ! CHECK-DAG: fir.global @_QMmEalloc_array : !fir.box<!fir.heap<!fir.array<?xi32>>>

  ! Non-array globals should not get alignment
  integer :: scalar_var
  ! CHECK-DAG: fir.global @_QMmEscalar_var : i32

  ! BIND(C) globals should not get alignment (C ABI)
  integer, bind(c, name="c_int_array") :: bind_c_int_array(10)
  ! CHECK-DAG: fir.global common @c_int_array : !fir.array<10xi32>
  real, bind(c, name="c_real_array") :: bind_c_real_array(5)
  ! CHECK-DAG: fir.global common @c_real_array : !fir.array<5xf32>

  ! BIND(C) arrays with initializers (exercises tryCreatingDenseGlobal path)
  integer, bind(c, name="c_init_array") :: bind_c_init_array(5) = [1,2,3,4,5]
  ! CHECK-DAG: fir.global @c_init_array(dense<[1, 2, 3, 4, 5]> : tensor<5xi32>) : !fir.array<5xi32>
  real, bind(c, name="c_real_init") :: bind_c_real_init(3) = [1.0, 2.0, 3.0]
  ! CHECK-DAG: fir.global @c_real_init(dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>) : !fir.array<3xf32>

  integer, bind(c, name="c_scalar") :: bind_c_scalar
  ! CHECK-DAG: fir.global common @c_scalar : i32
end module

subroutine sub_with_common()
  implicit none
  ! Common block (alignment from semantics, not 64)
  integer :: cb_int(10)
  real :: cb_real
  common /myblock/ cb_int, cb_real
  ! CHECK-DAG: fir.global common @myblock_(dense<0> : vector<44xi8>) {alignment = 4 : i64} : !fir.array<44xi8>
end subroutine

block data
  implicit none
  integer :: bd_array(5)
  common /initblock/ bd_array
  data bd_array /1, 2, 3, 4, 5/
  ! CHECK-DAG: fir.global @initblock_ {alignment = 4 : i64} : tuple<!fir.array<5xi32>>
end block data
