! RUN: bbc -polymorphic-type -emit-fir %s -o - | FileCheck %s
! RUN: bbc -polymorphic-type -emit-fir %s -o - | tco | FileCheck %s --check-prefix=LLVM

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
end module

program test_allocatable
  use poly

  class(p1), allocatable :: p
  class(p1), allocatable :: c1, c2
  class(p1), allocatable, dimension(:) :: c3, c4

  allocate(p) ! allocate as p1

  allocate(p1::c1)
  allocate(p2::c2)

  allocate(p1::c3(10))
  allocate(p2::c4(20))

  call c1%proc1()
  call c2%proc1()
end

! CHECK-LABEL: func.func @_QQmain()

! CHECK-DAG: %[[C1:.*]] = fir.address_of(@_QFEc1) : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK-DAG: %[[C2:.*]] = fir.address_of(@_QFEc2) : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK-DAG: %[[C3:.*]] = fir.address_of(@_QFEc3) : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK-DAG: %[[C4:.*]] = fir.address_of(@_QFEc4) : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK-DAG: %[[P:.*]] = fir.address_of(@_QFEp) : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[P_CAST:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerived(%[[P_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[C0]]) : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[P_CAST:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[P_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[C1_CAST:.*]] = fir.convert %0 : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerived(%[[C1_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[C0]]) : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C1_CAST:.*]] = fir.convert %[[C1]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[C1_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P2:.*]] = fir.address_of(@_QMpolyE.dt.p2) : !fir.ref<!fir.type<
! CHECK: %[[C2_CAST:.*]] = fir.convert %[[C2]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P2_CAST:.*]] = fir.convert %[[TYPE_DESC_P2]] : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerived(%[[C2_CAST]], %[[TYPE_DESC_P2_CAST]], %[[RANK]], %[[C0]]) : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C2_CAST:.*]] = fir.convert %[[C2]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[C2_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[RANK:.*]] = arith.constant 1 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerived(%[[C3_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[C0]]) : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C10:.*]] = arith.constant 10 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[C1_I64:.*]] = fir.convert %c1 : (index) -> i64
! CHECK: %[[C10_I64:.*]] = fir.convert %[[C10]] : (i32) -> i64
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableSetBounds(%[[C3_CAST]], %[[C0]], %[[C1_I64]], %[[C10_I64]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[C3_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P2:.*]] = fir.address_of(@_QMpolyE.dt.p2) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P2_CAST:.*]] = fir.convert %[[TYPE_DESC_P2]] : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[RANK:.*]] = arith.constant 1 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerived(%[[C4_CAST]], %[[TYPE_DESC_P2_CAST]], %[[RANK]], %[[C0]]) : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C20:.*]] = arith.constant 20 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[C1_I64:.*]] = fir.convert %[[C1]] : (index) -> i64
! CHECK: %[[C20_I64:.*]] = fir.convert %[[C20]] : (i32) -> i64
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableSetBounds(%[[C4_CAST]], %[[C0]], %[[C1_I64]], %[[C20_I64]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[C4_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32


! Check code generation of allocate runtime calls for polymoprhic entities. This
! is done from Fortran so we don't have a file full of auto-generated type info
! in order to perform the checks.

! LLVM-LABEL: define void @_QQmain()
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerived(ptr @_QFEp, ptr @_QMpolyE.dt.p1, i32 0, i32 0)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr @_QFEp, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerived(ptr @_QFEc1, ptr @_QMpolyE.dt.p1, i32 0, i32 0)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr @_QFEc1, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerived(ptr @_QFEc2, ptr @_QMpolyE.dt.p2, i32 0, i32 0)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr @_QFEc2, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerived(ptr @_QFEc3, ptr @_QMpolyE.dt.p1, i32 1, i32 0)
! LLVM: %{{.*}} = call {} @_FortranAAllocatableSetBounds(ptr @_QFEc3, i32 0, i64 1, i64 10)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr @_QFEc3, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerived(ptr @_QFEc4, ptr @_QMpolyE.dt.p2, i32 1, i32 0)
! LLVM: %{{.*}} = call {} @_FortranAAllocatableSetBounds(ptr @_QFEc4, i32 0, i64 1, i64 20)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr @_QFEc4, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
