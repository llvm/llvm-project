! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

! Test lowering of the initialization of coarrays.

! CHECK-LABEL: func.func @_QPtest_scalar()
subroutine test_scalar()
  integer, save :: x[*] = 1
! CHECK: %[[ADDR:.*]] = fir.address_of(@_QFtest_scalarEx) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]] uniq_name("_QFtest_scalarEx")
! CHECK: %[[C1:.*]] = arith.constant 1 : i32
! CHECK: %[[BOX:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK-NEXT: %[[HEAP:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<i32>, corank:1>) -> !fir.heap<i32>
! CHECK-NEXT: hlfir.assign %[[C1]] to %[[HEAP]] : i32, !fir.heap<i32>
! CHECK-NOT: fir.store
! CHECK: return
end subroutine
 
! CHECK-LABEL: func.func @_QPtest_real_kind8()
subroutine test_real_kind8()
  real(8), save :: x[*] = 2.5_8
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFtest_real_kind8Ex")
! CHECK: %[[CST:.*]] = arith.constant 2.500000e+00 : f64
! CHECK: %[[BOX:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<f64>, corank:1>>
! CHECK-NEXT: %[[HEAP:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<f64>, corank:1>) -> !fir.heap<f64>
! CHECK-NEXT: hlfir.assign %[[CST]] to %[[HEAP]] : f64, !fir.heap<f64>
end subroutine
 
! CHECK-LABEL: func.func @_QPtest_logical()
subroutine test_logical()
  logical, save :: x[*] = .true.
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFtest_logicalEx")
! CHECK: %[[TRUE:.*]] = arith.constant true
! CHECK: %[[BOX:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.logical<4>>, corank:1>>
! CHECK-NEXT: %[[HEAP:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.logical<4>>, corank:1>) -> !fir.heap<!fir.logical<4>>
! CHECK-NEXT: hlfir.assign %[[TRUE]] to %[[HEAP]] : i1, !fir.heap<!fir.logical<4>>
end subroutine
 
! CHECK-LABEL: func.func @_QPtest_complex()
subroutine test_complex()
  complex, save :: x[*] = (1.0, -1.0)
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFtest_complexEx")
! CHECK: %[[UNDEF:.*]] = fir.undefined complex<f32>
! CHECK: %[[RE:.*]] = fir.insert_value %[[UNDEF]], %{{.*}}, [0 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK: %[[CPLX:.*]] = fir.insert_value %[[RE]], %{{.*}}, [1 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK: %[[BOX:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<complex<f32>>, corank:1>>
! CHECK-NEXT: %[[HEAP:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<complex<f32>>, corank:1>) -> !fir.heap<complex<f32>>
! CHECK-NEXT: hlfir.assign %[[CPLX]] to %[[HEAP]] : complex<f32>, !fir.heap<complex<f32>>
end subroutine
 
! CHECK-LABEL: func.func @_QPtest_character()
subroutine test_character()
  character(len=5), save :: x[*] = "hello"
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %{{.*}} typeparams %{{.*}} uniq_name("_QFtest_characterEx")
! CHECK: %[[ADDR:.*]] = fir.address_of(@_QQclX68656C6C6F) : !fir.ref<!fir.char<1,5>>
! CHECK: %[[CST:.*]]:2 = hlfir.declare %[[ADDR]] typeparams %{{.*}} uniq_name("_QQclX68656C6C6F") fortran_attrs<parameter>
! CHECK: %[[BOX:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,5>>, corank:1>>
! CHECK-NEXT: %[[HEAP:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.char<1,5>>, corank:1>) -> !fir.heap<!fir.char<1,5>>
! CHECK-NEXT: hlfir.assign %[[CST]]#0 to %[[HEAP]] : !fir.ref<!fir.char<1,5>>, !fir.heap<!fir.char<1,5>>
end subroutine
 
! CHECK-LABEL: func.func @_QPtest_array()
subroutine test_array()
  real, save :: x(3)[*] = [1.0, 2.0, 3.0]
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFtest_arrayEx")
! CHECK: %[[ADDR:.*]] = fir.address_of(@_QQro.3xr4.{{.*}}) : !fir.ref<!fir.array<3xf32>>
! CHECK: %[[CST:.*]]:2 = hlfir.declare %[[ADDR]](%{{.*}}) uniq_name("_QQro.3xr4.{{.*}}") fortran_attrs<parameter>
! CHECK: %[[BOX:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<3xf32>>, corank:1>>
! CHECK-NEXT: hlfir.assign %[[CST]]#0 to %[[BOX]] : !fir.ref<!fir.array<3xf32>>, !fir.box<!fir.heap<!fir.array<3xf32>>, corank:1>
end subroutine
 
! CHECK-LABEL: func.func @_QPtest_array_broadcast()
subroutine test_array_broadcast()
  integer, save :: x(2, 3)[*] = 7
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFtest_array_broadcastEx")
! CHECK: %[[ADDR:.*]] = fir.address_of(@_QQro.2x3xi4.{{.*}}) : !fir.ref<!fir.array<2x3xi32>>
! CHECK: %[[CST:.*]]:2 = hlfir.declare %[[ADDR]](%{{.*}}) uniq_name("_QQro.2x3xi4.{{.*}}") fortran_attrs<parameter>
! CHECK: %[[BOX:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<2x3xi32>>, corank:1>>
! CHECK-NEXT: hlfir.assign %[[CST]]#0 to %[[BOX]] : !fir.ref<!fir.array<2x3xi32>>, !fir.box<!fir.heap<!fir.array<2x3xi32>>, corank:1>
end subroutine
 
! CHECK-LABEL: func.func @_QPtest_corank2()
subroutine test_corank2()
  integer, save :: x[2, *] = 3
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFtest_corank2Ex") : (!fir.ref<!fir.box<!fir.heap<i32>, corank:2>>)
! CHECK: %[[C3:.*]] = arith.constant 3 : i32
! CHECK: %[[BOX:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<i32>, corank:2>>
! CHECK-NEXT: %[[HEAP:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<i32>, corank:2>) -> !fir.heap<i32>
! CHECK-NEXT: hlfir.assign %[[C3]] to %[[HEAP]] : i32, !fir.heap<i32>
end subroutine
 
! CHECK-LABEL: func.func @_QPtest_data_stmt()
subroutine test_data_stmt()
  integer, save :: x[*]
  data x /5/
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFtest_data_stmtEx")
! CHECK: %[[C5:.*]] = arith.constant 5 : i32
! CHECK: %[[BOX:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK-NEXT: %[[HEAP:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<i32>, corank:1>) -> !fir.heap<i32>
! CHECK-NEXT: hlfir.assign %[[C5]] to %[[HEAP]] : i32, !fir.heap<i32>
end subroutine
 
! CHECK-LABEL: func.func @_QPtest_derived_explicit()
subroutine test_derived_explicit()
  type :: t
    integer :: i
    real :: r
  end type
  type(t), save :: x[*] = t(1, 2.0)
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFtest_derived_explicitEx")
! CHECK: %[[ADDR:.*]] = fir.address_of(@_QQro._QFtest_derived_explicitTt.{{.*}}) : !fir.ref<!fir.type<_QFtest_derived_explicitTt{i:i32,r:f32}>>
! CHECK: %[[CST:.*]]:2 = hlfir.declare %[[ADDR]] uniq_name("_QQro._QFtest_derived_explicitTt.{{.*}}") fortran_attrs<parameter>
! CHECK: %[[BOX:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_derived_explicitTt{i:i32,r:f32}>>, corank:1>>
! CHECK-NEXT: %[[HEAP:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.type<_QFtest_derived_explicitTt{i:i32,r:f32}>>, corank:1>) -> !fir.heap<!fir.type<_QFtest_derived_explicitTt{i:i32,r:f32}>>
! CHECK-NEXT: hlfir.assign %[[CST]]#0 to %[[HEAP]] : !fir.ref<!fir.type<_QFtest_derived_explicitTt{i:i32,r:f32}>>, !fir.heap<!fir.type<_QFtest_derived_explicitTt{i:i32,r:f32}>>
end subroutine
 
! No initializer: no assignment must be emitted.
! CHECK-LABEL: func.func @_QPtest_no_init()
subroutine test_no_init()
  integer, save :: x[*]
! CHECK: hlfir.declare %{{.*}} uniq_name("_QFtest_no_initEx")
! CHECK-NOT: hlfir.assign
! CHECK: return
end subroutine
 
! In the main program, the coarray is allocated with mif.alloc_coarray and then
! initialized.
! CHECK-LABEL: func.func @_QQmain()
program test_main
  integer :: x[*] = 100
! CHECK: %[[ADDR:.*]] = fir.address_of(@_QFEx) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK: mif.alloc_coarray %[[ADDR]] lcobounds %{{.*}} ucobounds %{{.*}} uniq_name("_QFEx")
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]] uniq_name("_QFEx")
! CHECK: %[[C100:.*]] = arith.constant 100 : i32
! CHECK: %[[BOX:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK-NEXT: %[[HEAP:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<i32>, corank:1>) -> !fir.heap<i32>
! CHECK-NEXT: hlfir.assign %[[C100]] to %[[HEAP]] : i32, !fir.heap<i32>
end program
