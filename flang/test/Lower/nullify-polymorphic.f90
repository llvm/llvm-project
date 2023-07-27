! RUN: bbc -polymorphic-type -emit-fir %s -o - | FileCheck %s

module poly
  type p1
    integer :: a
    integer :: b
  contains
    procedure, nopass :: proc1 => proc1_p1
  end type

  type, extends(p1) :: p2
    integer :: c
  contains
    procedure, nopass :: proc1 => proc1_p2
  end type
  
contains

  subroutine proc1_p1()
    print*, 'call proc1_p1'
  end subroutine

  subroutine proc1_p2()
    print*, 'call proc1_p2'
  end subroutine
  
  subroutine test_nullify()
    class(p1), pointer :: c

    allocate(p2::c)
    call c%proc1()

    nullify(c) ! c dynamic type must be reset to p1
  
    call c%proc1()
  end subroutine
end module

program test
  use poly
  call test_nullify()
end

! CHECK-LABEL: func.func @_QMpolyPtest_nullify()
! CHECK: %[[C_DESC:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>> {bindc_name = "c", uniq_name = "_QMpolyFtest_nullifyEc"}
! CHECK: %{{.*}} = fir.call @_FortranAPointerAllocate(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK: %[[DECLARED_TYPE_DESC:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}> 
! CHECK: %[[C_DESC_CAST:.*]] = fir.convert %[[C_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_CAST:.*]] = fir.convert %[[DECLARED_TYPE_DESC]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[CORANK:.*]] = arith.constant 0 : i32
! CHECK: %{{.*}} = fir.call @_FortranAPointerNullifyDerived(%[[C_DESC_CAST]], %[[TYPE_DESC_CAST]], %[[RANK]], %[[CORANK]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
