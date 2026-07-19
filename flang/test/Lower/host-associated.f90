! Test internal procedure host association lowering.
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! -----------------------------------------------------------------------------
!     Test non character intrinsic scalars
! -----------------------------------------------------------------------------

!!! Test scalar (with implicit none)

! CHECK-LABEL: func.func @_QPtest1() {
subroutine test1
  implicit none
  integer i
  ! CHECK: %[[i_alloca:.*]] = fir.alloca i32 {{.*}}uniq_name = "_QFtest1Ei"
  ! CHECK: %[[i:.*]]:2 = hlfir.declare %[[i_alloca]] {{.*}}uniq_name = "_QFtest1Ei"
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.ref<i32>>
  ! CHECK: fir.store %[[i]]#0 to %{{.*}} : !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK: fir.call @_QFtest1Ptest1_internal(%[[tup]])
  call test1_internal
  print *, i
contains
  ! CHECK-LABEL: func.func private @_QFtest1Ptest1_internal(
  ! CHECK-SAME: %[[arg:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc})
  ! CHECK: %[[i_ref:.*]] = fir.load %{{.*}} : !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK: %[[i:.*]]:2 = hlfir.declare %[[i_ref]] {{.*}}uniq_name = "_QFtest1Ei"
  ! CHECK: %[[val:.*]] = fir.call @_QPifoo()
  ! CHECK: hlfir.assign %[[val]] to %[[i]]#0 : i32, !fir.ref<i32>
  subroutine test1_internal
    integer, external :: ifoo
    i = ifoo()
  end subroutine test1_internal
end subroutine test1

!!! Test scalar

! CHECK-LABEL: func.func @_QPtest2() {
subroutine test2
  a = 1.0
  b = 2.0
  ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest2Ea"
  ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest2Eb"
  ! CHECK: fir.alloca tuple<!fir.ref<f32>, !fir.ref<f32>>
  ! CHECK: fir.call @_QFtest2Ptest2_internal
  call test2_internal
  print *, a, b
contains
  ! CHECK-LABEL: func.func private @_QFtest2Ptest2_internal(
  ! CHECK-SAME: %[[arg:.*]]: !fir.ref<tuple<!fir.ref<f32>, !fir.ref<f32>>> {fir.host_assoc})
  subroutine test2_internal
    ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest2Ea"
    ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest2Eb"
    c = a
    a = b
    b = c
    call test2_inner
  end subroutine test2_internal

  ! CHECK-LABEL: func.func private @_QFtest2Ptest2_inner(
  ! CHECK-SAME: %[[arg:.*]]: !fir.ref<tuple<!fir.ref<f32>, !fir.ref<f32>>> {fir.host_assoc})
  subroutine test2_inner
    ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest2Ea"
    ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest2Eb"
    if (a > b) then
       b = b + 2.0
    end if
  end subroutine test2_inner
end subroutine test2

! -----------------------------------------------------------------------------
!     Test character scalars
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest6(
subroutine test6(c)
  character(*) :: c
  ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest6Ec"
  ! CHECK: fir.alloca tuple<!fir.boxchar<1>>
  ! CHECK: fir.call @_QFtest6Ptest6_inner
  call test6_inner
  print *, c

contains
  ! CHECK-LABEL: func.func private @_QFtest6Ptest6_inner(
  ! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.boxchar<1>>> {fir.host_assoc})
  subroutine test6_inner
    ! CHECK: %[[load:.*]] = fir.load %{{.*}} : !fir.ref<!fir.boxchar<1>>
    ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest6Ec"
    c = "Hi there"
  end subroutine test6_inner
end subroutine test6

! -----------------------------------------------------------------------------
!     Test non allocatable and pointer arrays
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest3(
subroutine test3(p,q,i)
  integer(8) :: i
  real :: p(i:)
  real :: q(:)
  ! CHECK-DAG: hlfir.declare %{{.*}}uniq_name = "_QFtest3Ei"
  ! CHECK-DAG: hlfir.declare %{{.*}}uniq_name = "_QFtest3Ep"
  ! CHECK-DAG: hlfir.declare %{{.*}}uniq_name = "_QFtest3Eq"
  ! CHECK: fir.alloca tuple<!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>>
  ! CHECK: fir.call @_QFtest3Ptest3_inner
  call test3_inner
contains
  ! CHECK-LABEL: func.func private @_QFtest3Ptest3_inner(
  ! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>>> {fir.host_assoc})
  subroutine test3_inner
    ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest3Ep"
    ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest3Eq"
    p(2) = q(1)
  end subroutine test3_inner
end subroutine test3

! CHECK-LABEL: func.func @_QPtest3a(
subroutine test3a(p)
  real :: p(10)
  real :: q(10)
  ! CHECK-DAG: hlfir.declare %{{.*}}uniq_name = "_QFtest3aEp"
  ! CHECK-DAG: hlfir.declare %{{.*}}uniq_name = "_QFtest3aEq"
  ! CHECK: fir.alloca tuple<!fir.box<!fir.array<10xf32>>, !fir.box<!fir.array<10xf32>>>
  ! CHECK: fir.call @_QFtest3aPtest3a_inner
  call test3a_inner
contains
  ! CHECK-LABEL: func.func private @_QFtest3aPtest3a_inner(
  ! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.box<!fir.array<10xf32>>, !fir.box<!fir.array<10xf32>>>> {fir.host_assoc})
  subroutine test3a_inner
    ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest3aEp"
    ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest3aEq"
    p(1) = q(1)
  end subroutine test3a_inner
end subroutine test3a

! -----------------------------------------------------------------------------
!     Test allocatable and pointer scalars
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest4() {
subroutine test4
  real, pointer :: p
  real, allocatable, target :: ally
  ! CHECK-DAG: hlfir.declare %{{.*}}uniq_name = "_QFtest4Eally"
  ! CHECK-DAG: hlfir.declare %{{.*}}uniq_name = "_QFtest4Ep"
  ! CHECK: fir.alloca tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>
  ! CHECK: fir.call @_QFtest4Ptest4_inner
  allocate(ally)
  ally = -42.0
  call test4_inner
contains
  ! CHECK-LABEL: func.func private @_QFtest4Ptest4_inner(
  ! CHECK-SAME:%[[tup:.*]]: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>> {fir.host_assoc})
  subroutine test4_inner
    ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest4Ep"
    ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest4Eally"
    p => ally
  end subroutine test4_inner
end subroutine test4

! -----------------------------------------------------------------------------
!     Test allocatable and pointer arrays
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest5() {
subroutine test5
  real, pointer :: p(:)
  real, allocatable, target :: ally(:)
  ! CHECK-DAG: hlfir.declare %{{.*}}uniq_name = "_QFtest5Eally"
  ! CHECK-DAG: hlfir.declare %{{.*}}uniq_name = "_QFtest5Ep"
  ! CHECK: fir.alloca tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  ! CHECK: fir.call @_QFtest5Ptest5_inner
  allocate(ally(10))
  ally = -42.0
  call test5_inner
contains
  ! CHECK-LABEL: func.func private @_QFtest5Ptest5_inner(
  ! CHECK-SAME:%[[tup:.*]]: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>> {fir.host_assoc})
  subroutine test5_inner
    ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest5Ep"
    ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest5Eally"
    p => ally
  end subroutine test5_inner
end subroutine test5


! -----------------------------------------------------------------------------
!     Test elemental internal procedure
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest7(
subroutine test7(j, k)
  implicit none
  integer :: j
  integer :: k(:)
  ! CHECK-DAG: hlfir.declare %{{.*}}uniq_name = "_QFtest7Ej"
  ! CHECK-DAG: hlfir.declare %{{.*}}uniq_name = "_QFtest7Ek"
  ! CHECK: fir.alloca tuple<!fir.ref<i32>>
  ! CHECK: fir.call @_QFtest7Ptest7_inner
  k = test7_inner(k)
contains
! CHECK-LABEL: func.func private @_QFtest7Ptest7_inner(
! CHECK-SAME: %{{.*}}, %[[tup:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc})
elemental integer function test7_inner(i)
  implicit none
  integer, intent(in) :: i
  ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest7Ej"
  test7_inner = i + j
end function
end subroutine

subroutine issue990()
  implicit none
  integer :: captured
  call bar()
contains
! CHECK-LABEL: func.func private @_QFissue990Pbar(
! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc})
subroutine bar()
  integer :: stmt_func, i
  stmt_func(i) = i + captured
  ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFissue990Ecaptured"
  print *, stmt_func(10)
end subroutine
end subroutine

subroutine test8(dummy_proc)
 implicit none
 interface
   real function dummy_proc(x)
    real :: x
   end function
 end interface
 call bar()
contains
! CHECK-LABEL: func.func private @_QFtest8Pbar(
! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.boxproc<() -> ()>>> {fir.host_assoc})
subroutine bar()
  ! CHECK: fir.load %{{.*}} : !fir.ref<!fir.boxproc<() -> ()>>
 print *, dummy_proc(42.)
end subroutine
end subroutine

! CHECK-LABEL: func.func @_QPtest10(
subroutine test10(i)
 implicit none
 integer, pointer :: i(:)
 namelist /a_namelist/ i
 ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest10Ei"
 ! CHECK: fir.alloca tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>
 ! CHECK: fir.call @_QFtest10Pbar
 call bar()
contains
! CHECK-LABEL: func.func private @_QFtest10Pbar(
! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>> {fir.host_assoc})
subroutine bar()
  ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest10Ei"
  read (88, NML = a_namelist)
end subroutine
end subroutine

! CHECK-LABEL: func.func @_QPtest_proc_dummy() {
subroutine test_proc_dummy
  integer i
  i = 1
  ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest_proc_dummyEi"
  ! CHECK: fir.alloca tuple<!fir.ref<i32>>
  ! CHECK: fir.emboxproc
  call test_proc_dummy_other(test_proc_dummy_a)
  print *, i
contains
  ! CHECK-LABEL: func.func private @_QFtest_proc_dummyPtest_proc_dummy_a(
  ! CHECK-SAME: %{{.*}}, %[[tup:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc})
  subroutine test_proc_dummy_a(j)
    ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest_proc_dummyEi"
    i = i + j
  end subroutine test_proc_dummy_a
end subroutine test_proc_dummy

subroutine test_proc_dummy_char
  character(40) get_message
  external get_message
  character(10) message
  message = "Hi there!"
  ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest_proc_dummy_charEmessage"
  ! CHECK: fir.alloca tuple<!fir.boxchar<1>>
  ! CHECK: fir.emboxproc
  print *, get_message(gen_message)
contains
  ! CHECK-LABEL: func.func private @_QFtest_proc_dummy_charPgen_message(
  ! CHECK-SAME: %{{.*}}, %{{.*}}, %[[tup:.*]]: !fir.ref<tuple<!fir.boxchar<1>>> {fir.host_assoc})
  function gen_message
    ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest_proc_dummy_charEmessage"
    character(10) :: gen_message
    gen_message = message
  end function gen_message
end subroutine test_proc_dummy_char

! CHECK-LABEL: func.func @_QPtest_pdt_with_init_do_not_crash_host_symbol_analysis() {
subroutine test_pdt_with_init_do_not_crash_host_symbol_analysis()
  integer :: i
  call sub()
contains
  ! CHECK-LABEL: func.func private @_QFtest_pdt_with_init_do_not_crash_host_symbol_analysisPsub(
  ! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc})
  subroutine sub()
    ! CHECK: hlfir.declare %{{.*}}uniq_name = "_QFtest_pdt_with_init_do_not_crash_host_symbol_analysisEi"
    type type1 (k)
      integer, KIND :: k
      integer :: x = k
    end type
    type type2 (k, l)
      integer, KIND :: k = 4
      integer, LEN :: l = 2
      integer :: x = 10
      real :: y = 20
    end type
    print *, i
  end subroutine
end subroutine
