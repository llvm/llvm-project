! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

! Test lowering of the initialization of coarrays.

! CHECK-LABEL: func.func @_QPtest_scalar()
subroutine test_scalar()
  integer, save :: x[*] = 1
! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QFtest_scalarEx) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFtest_scalarEx")
! CHECK-NOT: hlfir.assign
! CHECK: return
end subroutine

! CHECK-LABEL: func.func @_QPtest_real_kind8()
subroutine test_real_kind8()
  real(8), save :: x[*] = 2.5_8
! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QFtest_real_kind8Ex) : !fir.ref<!fir.box<!fir.heap<f64>, corank:1>>
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFtest_real_kind8Ex")
! CHECK-NOT: hlfir.assign
! CHECK: return
end subroutine

! CHECK-LABEL: func.func @_QPtest_logical()
subroutine test_logical()
  logical, save :: x[*] = .true.
! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QFtest_logicalEx) : !fir.ref<!fir.box<!fir.heap<!fir.logical<4>>, corank:1>>
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFtest_logicalEx")
! CHECK-NOT: hlfir.assign
! CHECK: return
end subroutine

! CHECK-LABEL: func.func @_QPtest_complex()
subroutine test_complex()
  complex, save :: x[*] = (1.0, -1.0)
! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QFtest_complexEx) : !fir.ref<!fir.box<!fir.heap<complex<f32>>, corank:1>>
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFtest_complexEx")
! CHECK-NOT: hlfir.assign
! CHECK: return
end subroutine

! CHECK-LABEL: func.func @_QPtest_character()
subroutine test_character()
  character(len=5), save :: x[*] = "hello"
! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QFtest_characterEx) : !fir.ref<!fir.box<!fir.heap<!fir.char<1,5>>, corank:1>>
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] typeparams %{{.*}} uniq_name("_QFtest_characterEx")
! CHECK-NOT: hlfir.assign
! CHECK: return
end subroutine

! CHECK-LABEL: func.func @_QPtest_array()
subroutine test_array()
  real, save :: x(3)[*] = [1.0, 2.0, 3.0]
! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QFtest_arrayEx) : !fir.ref<!fir.box<!fir.heap<!fir.array<3xf32>>, corank:1>>
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFtest_arrayEx")
! CHECK-NOT: hlfir.assign
! CHECK: return
end subroutine

! CHECK-LABEL: func.func @_QPtest_array_broadcast()
subroutine test_array_broadcast()
  integer, save :: x(2, 3)[*] = 7
! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QFtest_array_broadcastEx) : !fir.ref<!fir.box<!fir.heap<!fir.array<2x3xi32>>, corank:1>>
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFtest_array_broadcastEx")
! CHECK-NOT: hlfir.assign
! CHECK: return
end subroutine

! CHECK-LABEL: func.func @_QPtest_corank2()
subroutine test_corank2()
  integer, save :: x[2, *] = 3
! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QFtest_corank2Ex) : !fir.ref<!fir.box<!fir.heap<i32>, corank:2>>
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFtest_corank2Ex")
! CHECK-NOT: hlfir.assign
! CHECK: return
end subroutine

! CHECK-LABEL: func.func @_QPtest_data_stmt()
subroutine test_data_stmt()
  integer, save :: x[*]
  data x /5/
! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QFtest_data_stmtEx) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFtest_data_stmtEx")
! CHECK-NOT: hlfir.assign
! CHECK: return
end subroutine

! CHECK-LABEL: func.func @_QPtest_derived_explicit()
subroutine test_derived_explicit()
  type :: t
    integer :: i
    real :: r
  end type
  type(t), save :: x[*] = t(1, 2.0)
! CHECK: %[[VAL_7:.*]] = fir.address_of(@_QFtest_derived_explicitEx) : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_derived_explicitTt{i:i32,r:f32}>>, corank:1>>
! CHECK: %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] uniq_name("_QFtest_derived_explicitEx")
! CHECK-NOT: hlfir.assign
! CHECK: return
end subroutine

! CHECK-LABEL: func.func @_QPtest_no_init()
subroutine test_no_init()
  integer, save :: x[*]
! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QFtest_no_initEx) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFtest_no_initEx")
! CHECK-NOT: hlfir.assign
! CHECK: return
end subroutine

! CHECK-LABEL: func.func @_QQmain()
program test_main
  integer :: x[*] = 100
! CHECK: %[[VAL_3:.*]] = fir.address_of(@_QFEx) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK: %[[VAL_5:.*]] = fir.embox %{{.*}} : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
! CHECK: %[[VAL_6:.*]] = fir.embox %{{.*}} : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
! CHECK: mif.alloc_coarray %[[VAL_3]] lcobounds %[[VAL_5]] ucobounds %[[VAL_6]] uniq_name("_QFEx")
! CHECK: %[[C100_I32:.*]] = arith.constant 100 : i32
! CHECK: %[[VAL_7:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK-NEXT: %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.heap<i32>, corank:1>) -> !fir.heap<i32>
! CHECK-NEXT: hlfir.assign %[[C100_I32]] to %[[VAL_8]] : i32, !fir.heap<i32>
! CHECK: %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QFEx")
end program

! The saved coarrays of the subroutines are allocated and initialized in @__mif_save_coarrays_allocate.
! CHECK-LABEL: func.func @__mif_save_coarrays_allocate()
! CHECK: %[[VAL_22:.*]] = mif.init -> i32

! CHECK: %[[VAL_23:.*]] = fir.address_of(@_QFtest_scalarEx) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK: mif.alloc_coarray %[[VAL_23]] lcobounds %{{.*}} ucobounds %{{.*}} uniq_name("_QFtest_scalarEx")
! CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[VAL_27:.*]] = fir.load %[[VAL_23]] : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK-NEXT: %[[VAL_28:.*]] = fir.box_addr %[[VAL_27]] : (!fir.box<!fir.heap<i32>, corank:1>) -> !fir.heap<i32>
! CHECK-NEXT: hlfir.assign %[[C1_I32]] to %[[VAL_28]] : i32, !fir.heap<i32>

! CHECK: %[[VAL_29:.*]] = fir.address_of(@_QFtest_real_kind8Ex) : !fir.ref<!fir.box<!fir.heap<f64>, corank:1>>
! CHECK: mif.alloc_coarray %[[VAL_29]] lcobounds %{{.*}} ucobounds %{{.*}} uniq_name("_QFtest_real_kind8Ex")
! CHECK: %[[CST:.*]] = arith.constant 2.500000e+00 : f64
! CHECK: %[[VAL_33:.*]] = fir.load %[[VAL_29]] : !fir.ref<!fir.box<!fir.heap<f64>, corank:1>>
! CHECK-NEXT: %[[VAL_34:.*]] = fir.box_addr %[[VAL_33]] : (!fir.box<!fir.heap<f64>, corank:1>) -> !fir.heap<f64>
! CHECK-NEXT: hlfir.assign %[[CST]] to %[[VAL_34]] : f64, !fir.heap<f64>

! CHECK: %[[VAL_35:.*]] = fir.address_of(@_QFtest_logicalEx) : !fir.ref<!fir.box<!fir.heap<!fir.logical<4>>, corank:1>>
! CHECK: mif.alloc_coarray %[[VAL_35]] lcobounds %{{.*}} ucobounds %{{.*}} uniq_name("_QFtest_logicalEx")
! CHECK: %[[TRUE:.*]] = arith.constant true
! CHECK: %[[VAL_39:.*]] = fir.load %[[VAL_35]] : !fir.ref<!fir.box<!fir.heap<!fir.logical<4>>, corank:1>>
! CHECK-NEXT: %[[VAL_40:.*]] = fir.box_addr %[[VAL_39]] : (!fir.box<!fir.heap<!fir.logical<4>>, corank:1>) -> !fir.heap<!fir.logical<4>>
! CHECK-NEXT: hlfir.assign %[[TRUE]] to %[[VAL_40]] : i1, !fir.heap<!fir.logical<4>>

! CHECK: %[[VAL_41:.*]] = fir.address_of(@_QFtest_complexEx) : !fir.ref<!fir.box<!fir.heap<complex<f32>>, corank:1>>
! CHECK: mif.alloc_coarray %[[VAL_41]] lcobounds %{{.*}} ucobounds %{{.*}} uniq_name("_QFtest_complexEx")
! CHECK: %[[CST_14:.*]] = arith.constant 1.000000e+00 : f32
! CHECK: %[[CST_15:.*]] = arith.constant -1.000000e+00 : f32
! CHECK: %[[VAL_45:.*]] = fir.undefined complex<f32>
! CHECK: %[[VAL_46:.*]] = fir.insert_value %[[VAL_45]], %[[CST_14]], [0 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK: %[[VAL_47:.*]] = fir.insert_value %[[VAL_46]], %[[CST_15]], [1 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK: %[[VAL_48:.*]] = fir.load %[[VAL_41]] : !fir.ref<!fir.box<!fir.heap<complex<f32>>, corank:1>>
! CHECK-NEXT: %[[VAL_49:.*]] = fir.box_addr %[[VAL_48]] : (!fir.box<!fir.heap<complex<f32>>, corank:1>) -> !fir.heap<complex<f32>>
! CHECK-NEXT: hlfir.assign %[[VAL_47]] to %[[VAL_49]] : complex<f32>, !fir.heap<complex<f32>>

! CHECK: %[[VAL_50:.*]] = fir.address_of(@_QFtest_characterEx) : !fir.ref<!fir.box<!fir.heap<!fir.char<1,5>>, corank:1>>
! CHECK: mif.alloc_coarray %[[VAL_50]] lcobounds %{{.*}} ucobounds %{{.*}} uniq_name("_QFtest_characterEx")
! CHECK: %[[VAL_54:.*]] = fir.address_of(@_QQclX68656C6C6F) : !fir.ref<!fir.char<1,5>>
! CHECK: %[[VAL_55:.*]]:2 = hlfir.declare %[[VAL_54]] typeparams %{{.*}} uniq_name("_QQclX68656C6C6F") fortran_attrs<parameter>
! CHECK: %[[VAL_56:.*]] = fir.load %[[VAL_50]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,5>>, corank:1>>
! CHECK-NEXT: %[[VAL_57:.*]] = fir.box_addr %[[VAL_56]] : (!fir.box<!fir.heap<!fir.char<1,5>>, corank:1>) -> !fir.heap<!fir.char<1,5>>
! CHECK-NEXT: hlfir.assign %[[VAL_55]]#0 to %[[VAL_57]] : !fir.ref<!fir.char<1,5>>, !fir.heap<!fir.char<1,5>>

! CHECK: %[[VAL_58:.*]] = fir.address_of(@_QFtest_arrayEx) : !fir.ref<!fir.box<!fir.heap<!fir.array<3xf32>>, corank:1>>
! CHECK: mif.alloc_coarray %[[VAL_58]] lcobounds %{{.*}} ucobounds %{{.*}} uniq_name("_QFtest_arrayEx")
! CHECK: %[[VAL_62:.*]] = fir.address_of(@_QQro.3xr4.{{.*}}) : !fir.ref<!fir.array<3xf32>>
! CHECK: %[[VAL_64:.*]]:2 = hlfir.declare %[[VAL_62]](%{{.*}}) uniq_name("_QQro.3xr4.{{.*}}") fortran_attrs<parameter>
! CHECK: %[[VAL_65:.*]] = fir.load %[[VAL_58]] : !fir.ref<!fir.box<!fir.heap<!fir.array<3xf32>>, corank:1>>
! CHECK-NEXT: hlfir.assign %[[VAL_64]]#0 to %[[VAL_65]] : !fir.ref<!fir.array<3xf32>>, !fir.box<!fir.heap<!fir.array<3xf32>>, corank:1>

! CHECK: %[[VAL_66:.*]] = fir.address_of(@_QFtest_array_broadcastEx) : !fir.ref<!fir.box<!fir.heap<!fir.array<2x3xi32>>, corank:1>>
! CHECK: mif.alloc_coarray %[[VAL_66]] lcobounds %{{.*}} ucobounds %{{.*}} uniq_name("_QFtest_array_broadcastEx")
! CHECK: %[[VAL_70:.*]] = fir.address_of(@_QQro.2x3xi4.{{.*}}) : !fir.ref<!fir.array<2x3xi32>>
! CHECK: %[[VAL_72:.*]]:2 = hlfir.declare %[[VAL_70]](%{{.*}}) uniq_name("_QQro.2x3xi4.{{.*}}") fortran_attrs<parameter>
! CHECK: %[[VAL_73:.*]] = fir.load %[[VAL_66]] : !fir.ref<!fir.box<!fir.heap<!fir.array<2x3xi32>>, corank:1>>
! CHECK-NEXT: hlfir.assign %[[VAL_72]]#0 to %[[VAL_73]] : !fir.ref<!fir.array<2x3xi32>>, !fir.box<!fir.heap<!fir.array<2x3xi32>>, corank:1>

! CHECK: %[[VAL_74:.*]] = fir.address_of(@_QFtest_corank2Ex) : !fir.ref<!fir.box<!fir.heap<i32>, corank:2>>
! CHECK: mif.alloc_coarray %[[VAL_74]] lcobounds %{{.*}} ucobounds %{{.*}} uniq_name("_QFtest_corank2Ex")
! CHECK: %[[C3_I32:.*]] = arith.constant 3 : i32
! CHECK: %[[VAL_80:.*]] = fir.load %[[VAL_74]] : !fir.ref<!fir.box<!fir.heap<i32>, corank:2>>
! CHECK-NEXT: %[[VAL_81:.*]] = fir.box_addr %[[VAL_80]] : (!fir.box<!fir.heap<i32>, corank:2>) -> !fir.heap<i32>
! CHECK-NEXT: hlfir.assign %[[C3_I32]] to %[[VAL_81]] : i32, !fir.heap<i32>

! CHECK: %[[VAL_82:.*]] = fir.address_of(@_QFtest_data_stmtEx) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK: mif.alloc_coarray %[[VAL_82]] lcobounds %{{.*}} ucobounds %{{.*}} uniq_name("_QFtest_data_stmtEx")
! CHECK: %[[C5_I32:.*]] = arith.constant 5 : i32
! CHECK: %[[VAL_86:.*]] = fir.load %[[VAL_82]] : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK-NEXT: %[[VAL_87:.*]] = fir.box_addr %[[VAL_86]] : (!fir.box<!fir.heap<i32>, corank:1>) -> !fir.heap<i32>
! CHECK-NEXT: hlfir.assign %[[C5_I32]] to %[[VAL_87]] : i32, !fir.heap<i32>

! CHECK: %[[VAL_88:.*]] = fir.address_of(@_QFtest_derived_explicitEx) : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_derived_explicitTt{i:i32,r:f32}>>, corank:1>>
! CHECK: mif.alloc_coarray %[[VAL_88]] lcobounds %{{.*}} ucobounds %{{.*}} uniq_name("_QFtest_derived_explicitEx")
! CHECK: %[[VAL_92:.*]] = fir.address_of(@_QQro._QFtest_derived_explicitTt.{{.*}}) : !fir.ref<!fir.type<_QFtest_derived_explicitTt{i:i32,r:f32}>>
! CHECK: %[[VAL_93:.*]]:2 = hlfir.declare %[[VAL_92]] uniq_name("_QQro._QFtest_derived_explicitTt.{{.*}}") fortran_attrs<parameter>
! CHECK: %[[VAL_94:.*]] = fir.load %[[VAL_88]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_derived_explicitTt{i:i32,r:f32}>>, corank:1>>
! CHECK-NEXT: %[[VAL_95:.*]] = fir.box_addr %[[VAL_94]] : (!fir.box<!fir.heap<!fir.type<_QFtest_derived_explicitTt{i:i32,r:f32}>>, corank:1>) -> !fir.heap<!fir.type<_QFtest_derived_explicitTt{i:i32,r:f32}>>
! CHECK-NEXT: hlfir.assign %[[VAL_93]]#0 to %[[VAL_95]] : !fir.ref<!fir.type<_QFtest_derived_explicitTt{i:i32,r:f32}>>, !fir.heap<!fir.type<_QFtest_derived_explicitTt{i:i32,r:f32}>>

! CHECK: %[[VAL_96:.*]] = fir.address_of(@_QFtest_no_initEx) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK: mif.alloc_coarray %[[VAL_96]] lcobounds %{{.*}} ucobounds %{{.*}} uniq_name("_QFtest_no_initEx")
! CHECK-NOT: hlfir.assign
! CHECK: return
