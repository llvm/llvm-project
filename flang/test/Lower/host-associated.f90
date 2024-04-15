! Test internal procedure host association lowering.
! RUN: bbc -hlfir=false %s -o - | FileCheck %s

! -----------------------------------------------------------------------------
!     Test non character intrinsic scalars
! -----------------------------------------------------------------------------

!!! Test scalar (with implicit none)

! CHECK-LABEL: func @_QPtest1(
subroutine test1
  implicit none
  integer i
  ! CHECK-DAG: %[[i:.*]] = fir.alloca i32 {{.*}}uniq_name = "_QFtest1Ei"
  ! CHECK-DAG: %[[tup:.*]] = fir.alloca tuple<!fir.ref<i32>>
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[tup]], %c0
  ! CHECK: fir.store %[[i]] to %[[addr]] : !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK: fir.call @_QFtest1Ptest1_internal(%[[tup]]) {{.*}}: (!fir.ref<tuple<!fir.ref<i32>>>) -> ()
  call test1_internal
  print *, i
contains
  ! CHECK-LABEL: func private @_QFtest1Ptest1_internal(
  ! CHECK-SAME: %[[arg:[^:]*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
  ! CHECK: %[[iaddr:.*]] = fir.coordinate_of %[[arg]], %c0
  ! CHECK: %[[i:.*]] = fir.load %[[iaddr]] : !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK: %[[val:.*]] = fir.call @_QPifoo() {{.*}}: () -> i32
  ! CHECK: fir.store %[[val]] to %[[i]] : !fir.ref<i32>
  subroutine test1_internal
    integer, external :: ifoo
    i = ifoo()
  end subroutine test1_internal
end subroutine test1

!!! Test scalar

! CHECK-LABEL: func @_QPtest2() {
subroutine test2
  a = 1.0
  b = 2.0
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.ref<f32>, !fir.ref<f32>>
  ! CHECK: %[[a0:.*]] = fir.coordinate_of %[[tup]], %c0
  ! CHECK: fir.store %{{.*}} to %[[a0]] : !fir.llvm_ptr<!fir.ref<f32>>
  ! CHECK: %[[b0:.*]] = fir.coordinate_of %[[tup]], %c1
  ! CHECK: fir.store %{{.*}} to %[[b0]] : !fir.llvm_ptr<!fir.ref<f32>>
  ! CHECK: fir.call @_QFtest2Ptest2_internal(%[[tup]]) {{.*}}: (!fir.ref<tuple<!fir.ref<f32>, !fir.ref<f32>>>) -> ()
  call test2_internal
  print *, a, b
contains
  ! CHECK-LABEL: func private @_QFtest2Ptest2_internal(
  ! CHECK-SAME: %[[arg:[^:]*]]: !fir.ref<tuple<!fir.ref<f32>, !fir.ref<f32>>> {fir.host_assoc}) attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
  subroutine test2_internal
    ! CHECK: %[[a:.*]] = fir.coordinate_of %[[arg]], %c0
    ! CHECK: %[[aa:.*]] = fir.load %[[a]] : !fir.llvm_ptr<!fir.ref<f32>>
    ! CHECK: %[[b:.*]] = fir.coordinate_of %[[arg]], %c1
    ! CHECK: %{{.*}} = fir.load %[[b]] : !fir.llvm_ptr<!fir.ref<f32>>
    ! CHECK: fir.alloca
    ! CHECK: fir.load %[[aa]] : !fir.ref<f32>
    c = a
    a = b
    b = c
    call test2_inner
  end subroutine test2_internal

  ! CHECK-LABEL: func private @_QFtest2Ptest2_inner(
  ! CHECK-SAME: %[[arg:[^:]*]]: !fir.ref<tuple<!fir.ref<f32>, !fir.ref<f32>>> {fir.host_assoc}) attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
  subroutine test2_inner
    ! CHECK: %[[a:.*]] = fir.coordinate_of %[[arg]], %c0
    ! CHECK: %[[aa:.*]] = fir.load %[[a]] : !fir.llvm_ptr<!fir.ref<f32>>
    ! CHECK: %[[b:.*]] = fir.coordinate_of %[[arg]], %c1
    ! CHECK: %[[bb:.*]] = fir.load %[[b]] : !fir.llvm_ptr<!fir.ref<f32>>
    ! CHECK-DAG: %[[bd:.*]] = fir.load %[[bb]] : !fir.ref<f32>
    ! CHECK-DAG: %[[ad:.*]] = fir.load %[[aa]] : !fir.ref<f32>
    ! CHECK: %{{.*}} = arith.cmpf ogt, %[[ad]], %[[bd]] {{.*}} : f32
    if (a > b) then
       b = b + 2.0
    end if
  end subroutine test2_inner
end subroutine test2

! -----------------------------------------------------------------------------
!     Test non character scalars
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest6(
! CHECK-SAME: %[[c:.*]]: !fir.boxchar<1>
subroutine test6(c)
  character(*) :: c
  ! CHECK: %[[cunbox:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.boxchar<1>>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
  ! CHECK: %[[emboxchar:.*]] = fir.emboxchar %[[cunbox]]#0, %[[cunbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.store %[[emboxchar]] to %[[coor]] : !fir.ref<!fir.boxchar<1>>
  ! CHECK: fir.call @_QFtest6Ptest6_inner(%[[tup]]) {{.*}}: (!fir.ref<tuple<!fir.boxchar<1>>>) -> ()
  call test6_inner
  print *, c

contains
  ! CHECK-LABEL: func private @_QFtest6Ptest6_inner(
  ! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.boxchar<1>>> {fir.host_assoc}) attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
  subroutine test6_inner
    ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
    ! CHECK: %[[load:.*]] = fir.load %[[coor]] : !fir.ref<!fir.boxchar<1>>
    ! CHECK: fir.unboxchar %[[load]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    c = "Hi there"
  end subroutine test6_inner
end subroutine test6

! -----------------------------------------------------------------------------
!     Test non allocatable and pointer arrays
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest3(
! CHECK-SAME: %[[p:[^:]+]]: !fir.box<!fir.array<?xf32>>{{.*}}, %[[q:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}, %[[i:.*]]: !fir.ref<i64>
subroutine test3(p,q,i)
  integer(8) :: i
  real :: p(i:)
  real :: q(:)
  ! CHECK: %[[iload:.*]] = fir.load %[[i]] : !fir.ref<i64>
  ! CHECK: %[[icast:.*]] = fir.convert %[[iload]] : (i64) -> index
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>>
  ! CHECK: %[[ptup:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<?xf32>>>
  ! CHECK: %[[pshift:.*]] = fir.shift %[[icast]] : (index) -> !fir.shift<1>
  ! CHECK: %[[pbox:.*]] = fir.rebox %[[p]](%[[pshift]]) : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: fir.store %[[pbox]] to %[[ptup]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
  ! CHECK: %[[qtup:.*]] = fir.coordinate_of %[[tup]], %c1{{.*}} : (!fir.ref<tuple<!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<?xf32>>>
  ! CHECK: %[[qbox:.*]] = fir.rebox %[[q]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: fir.store %[[qbox]] to %[[qtup]] : !fir.ref<!fir.box<!fir.array<?xf32>>>

  i = i + 1
  q = -42.0

  ! CHECK: fir.call @_QFtest3Ptest3_inner(%[[tup]]) {{.*}}: (!fir.ref<tuple<!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>>>) -> ()
  call test3_inner

  if (p(2) .ne. -42.0) then
     print *, "failed"
  end if
  
contains
  ! CHECK-LABEL: func private @_QFtest3Ptest3_inner(
  ! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>>> {fir.host_assoc}) attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
  subroutine test3_inner
    ! CHECK: %[[pcoor:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<?xf32>>>
    ! CHECK: %[[p:.*]] = fir.load %[[pcoor]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
    ! CHECK: %[[pbounds:.]]:3 = fir.box_dims %[[p]], %c0{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
    ! CHECK: %[[qcoor:.*]] = fir.coordinate_of %[[tup]], %c1{{.*}} : (!fir.ref<tuple<!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<?xf32>>>
    ! CHECK: %[[q:.*]] = fir.load %[[qcoor]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
    ! CHECK: %[[qbounds:.]]:3 = fir.box_dims %[[q]], %c0{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)


    ! CHECK: %[[qlb:.*]] = fir.convert %[[qbounds]]#0 : (index) -> i64
    ! CHECK: %[[qoffset:.*]] = arith.subi %c1{{.*}}, %[[qlb]] : i64
    ! CHECK: %[[qelt:.*]] = fir.coordinate_of %[[q]], %[[qoffset]] : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
    ! CHECK: %[[qload:.*]] = fir.load %[[qelt]] : !fir.ref<f32>
    ! CHECK: %[[plb:.*]] = fir.convert %[[pbounds]]#0 : (index) -> i64
    ! CHECK: %[[poffset:.*]] = arith.subi %c2{{.*}}, %[[plb]] : i64
    ! CHECK: %[[pelt:.*]] = fir.coordinate_of %[[p]], %[[poffset]] : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
    ! CHECK: fir.store %[[qload]] to %[[pelt]] : !fir.ref<f32>
    p(2) = q(1)
  end subroutine test3_inner
end subroutine test3

! CHECK-LABEL: func @_QPtest3a(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.array<10xf32>>{{.*}}) {
subroutine test3a(p)
  real :: p(10)
  real :: q(10)
  ! CHECK: %[[q:.*]] = fir.alloca !fir.array<10xf32> {bindc_name = "q", uniq_name = "_QFtest3aEq"}
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.box<!fir.array<10xf32>>, !fir.box<!fir.array<10xf32>>>
  ! CHECK: %[[ptup:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.box<!fir.array<10xf32>>, !fir.box<!fir.array<10xf32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<10xf32>>>
  ! CHECK: %[[shape:.*]] = fir.shape %c10{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[pbox:.*]] = fir.embox %[[p]](%[[shape]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
  ! CHECK: fir.store %[[pbox]] to %[[ptup]] : !fir.ref<!fir.box<!fir.array<10xf32>>>
  ! CHECK: %[[qtup:.*]] = fir.coordinate_of %[[tup]], %c1{{.*}} : (!fir.ref<tuple<!fir.box<!fir.array<10xf32>>, !fir.box<!fir.array<10xf32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<10xf32>>>
  ! CHECK: %[[qbox:.*]] = fir.embox %[[q]](%[[shape]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
  ! CHECK: fir.store %[[qbox]] to %[[qtup]] : !fir.ref<!fir.box<!fir.array<10xf32>>>

  q = -42.0
  ! CHECK: fir.call @_QFtest3aPtest3a_inner(%[[tup]]) {{.*}}: (!fir.ref<tuple<!fir.box<!fir.array<10xf32>>, !fir.box<!fir.array<10xf32>>>>) -> ()
  call test3a_inner

  if (p(1) .ne. -42.0) then
     print *, "failed"
  end if
  
contains
  ! CHECK: func private @_QFtest3aPtest3a_inner(
  ! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.box<!fir.array<10xf32>>, !fir.box<!fir.array<10xf32>>>> {fir.host_assoc}) attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
  subroutine test3a_inner
    ! CHECK: %[[pcoor:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.box<!fir.array<10xf32>>, !fir.box<!fir.array<10xf32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<10xf32>>>
    ! CHECK: %[[p:.*]] = fir.load %[[pcoor]] : !fir.ref<!fir.box<!fir.array<10xf32>>>
    ! CHECK: %[[paddr:.*]] = fir.box_addr %[[p]] : (!fir.box<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>>
    ! CHECK: %[[qcoor:.*]] = fir.coordinate_of %[[tup]], %c1{{.*}} : (!fir.ref<tuple<!fir.box<!fir.array<10xf32>>, !fir.box<!fir.array<10xf32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<10xf32>>>
    ! CHECK: %[[q:.*]] = fir.load %[[qcoor]] : !fir.ref<!fir.box<!fir.array<10xf32>>>
    ! CHECK: %[[qaddr:.*]] = fir.box_addr %[[q]] : (!fir.box<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>>

    ! CHECK: %[[qelt:.*]] = fir.coordinate_of %[[qaddr]], %c0{{.*}} : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
    ! CHECK: %[[qload:.*]] = fir.load %[[qelt]] : !fir.ref<f32>
    ! CHECK: %[[pelt:.*]] = fir.coordinate_of %[[paddr]], %c0{{.*}} : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
    ! CHECK: fir.store %[[qload]] to %[[pelt]] : !fir.ref<f32>
    p(1) = q(1)
  end subroutine test3a_inner
end subroutine test3a

! -----------------------------------------------------------------------------
!     Test allocatable and pointer scalars
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest4() {
subroutine test4
  real, pointer :: p
  real, allocatable, target :: ally
  ! CHECK: %[[ally:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "ally", fir.target, uniq_name = "_QFtest4Eally"}
  ! CHECK: %[[p:.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "p", uniq_name = "_QFtest4Ep"}
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>
  ! CHECK: %[[ptup:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<f32>>>>
  ! CHECK: fir.store %[[p]] to %[[ptup]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<f32>>>>
  ! CHECK: %[[atup:.*]] = fir.coordinate_of %[[tup]], %c1{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<f32>>>>
  ! CHECK: fir.store %[[ally]] to %[[atup]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<f32>>>>
  ! CHECK: fir.call @_QFtest4Ptest4_inner(%[[tup]]) {{.*}}: (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>>) -> ()

  allocate(ally)
  ally = -42.0
  call test4_inner

  if (p .ne. -42.0) then
     print *, "failed"
  end if
  
contains
  ! CHECK-LABEL: func private @_QFtest4Ptest4_inner(
  ! CHECK-SAME:%[[tup:.*]]: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>> {fir.host_assoc}) attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
  subroutine test4_inner
    ! CHECK: %[[ptup:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<f32>>>>
    ! CHECK: %[[p:.*]] = fir.load %[[ptup]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<f32>>>>
    ! CHECK: %[[atup:.*]] = fir.coordinate_of %[[tup]], %c1{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<f32>>>>
    ! CHECK: %[[a:.*]] = fir.load %[[atup]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<f32>>>>
    ! CHECK: %[[abox:.*]] = fir.load %[[a]] : !fir.ref<!fir.box<!fir.heap<f32>>>
    ! CHECK: %[[addr:.*]] = fir.box_addr %[[abox]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
    ! CHECK: %[[ptr:.*]] = fir.embox %[[addr]] : (!fir.heap<f32>) -> !fir.box<!fir.ptr<f32>>
    ! CHECK: fir.store %[[ptr]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
    p => ally
  end subroutine test4_inner
end subroutine test4

! -----------------------------------------------------------------------------
!     Test allocatable and pointer arrays
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest5() {
subroutine test5
  real, pointer :: p(:)
  real, allocatable, target :: ally(:)

  ! CHECK: %[[ally:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "ally", fir.target
  ! CHECK: %[[p:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>> {bindc_name = "p"
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  ! CHECK: %[[ptup:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>>
  ! CHECK: fir.store %[[p]] to %[[ptup]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>>
  ! CHECK: %[[atup:.*]] = fir.coordinate_of %[[tup]], %c1{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  ! CHECK: fir.store %[[ally]] to %[[atup]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  ! CHECK: fir.call @_QFtest5Ptest5_inner(%[[tup]]) {{.*}}: (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>) -> ()

  allocate(ally(10))
  ally = -42.0
  call test5_inner

  if (p(1) .ne. -42.0) then
     print *, "failed"
  end if
  
contains
  ! CHECK-LABEL: func private @_QFtest5Ptest5_inner(
  ! CHECK-SAME:%[[tup:.*]]: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>> {fir.host_assoc}) attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
  subroutine test5_inner
    ! CHECK: %[[ptup:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>>
    ! CHECK: %[[p:.*]] = fir.load %[[ptup]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>>
    ! CHECK: %[[atup:.*]] = fir.coordinate_of %[[tup]], %c1{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
    ! CHECK: %[[a:.*]] = fir.load %[[atup]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
    ! CHECK: %[[abox:.*]] = fir.load %[[a]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    ! CHECK-DAG: %[[adims:.*]]:3 = fir.box_dims %[[abox]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
    ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[abox]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
    ! CHECK-DAG: %[[ashape:.*]] = fir.shape_shift %[[adims]]#0, %[[adims]]#1 : (index, index) -> !fir.shapeshift<1>

    ! CHECK: %[[ptr:.*]] = fir.embox %[[addr]](%[[ashape]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
    ! CHECK: fir.store %[[ptr]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    p => ally
  end subroutine test5_inner
end subroutine test5


! -----------------------------------------------------------------------------
!     Test elemental internal procedure
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest7(
! CHECK-SAME: %[[j:.*]]: !fir.ref<i32>{{.*}}, %[[k:.*]]: !fir.box<!fir.array<?xi32>>
subroutine test7(j, k)
  implicit none
  integer :: j
  integer :: k(:)
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.ref<i32>>
  ! CHECK: %[[jtup:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK: fir.store %[[j]] to %[[jtup]] : !fir.llvm_ptr<!fir.ref<i32>>

  ! CHECK: %[[kelem:.*]] = fir.array_coor %[[k]] %{{.*}} : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
  ! CHECK: fir.call @_QFtest7Ptest7_inner(%[[kelem]], %[[tup]]) {{.*}}: (!fir.ref<i32>, !fir.ref<tuple<!fir.ref<i32>>>) -> i32
  k = test7_inner(k)
contains

! CHECK-LABEL: func private @_QFtest7Ptest7_inner(
! CHECK-SAME: %[[i:.*]]: !fir.ref<i32>{{.*}}, %[[tup:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) -> i32 attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
elemental integer function test7_inner(i)
  implicit none
  integer, intent(in) :: i
  ! CHECK: %[[jtup:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK: %[[jptr:.*]] = fir.load %[[jtup]] : !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK-DAG: %[[iload:.*]] = fir.load %[[i]] : !fir.ref<i32>
  ! CHECK-DAG: %[[jload:.*]] = fir.load %[[jptr]] : !fir.ref<i32>
  ! CHECK: addi %[[iload]], %[[jload]] : i32
  test7_inner = i + j
end function
end subroutine

subroutine issue990()
  ! Test that host symbols used in statement functions inside an internal
  ! procedure are correctly captured from the host.
  implicit none
  integer :: captured
  call bar()
contains
! CHECK-LABEL: func private @_QFissue990Pbar(
! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
subroutine bar()
  integer :: stmt_func, i
  stmt_func(i) = i + captured
  ! CHECK: %[[tupAddr:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK: %[[addr:.*]] = fir.load %[[tupAddr]] : !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK: %[[value:.*]] = fir.load %[[addr]] : !fir.ref<i32>
  ! CHECK: arith.addi %{{.*}}, %[[value]] : i32
  print *, stmt_func(10)
end subroutine
end subroutine

subroutine issue990b()
  ! Test when an internal procedure uses a statement function from its host
  ! which uses host variables that are otherwise not used by the internal
  ! procedure.
  implicit none
  integer :: captured, captured_stmt_func, i
  captured_stmt_func(i) = i + captured
  call bar()
contains
! CHECK-LABEL: func private @_QFissue990bPbar(
! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
subroutine bar()
  ! CHECK: %[[tupAddr:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK: %[[addr:.*]] = fir.load %[[tupAddr]] : !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK: %[[value:.*]] = fir.load %[[addr]] : !fir.ref<i32>
  ! CHECK: arith.addi %{{.*}}, %[[value]] : i32
  print *, captured_stmt_func(10)
end subroutine
end subroutine

! Test capture of dummy procedure functions.
subroutine test8(dummy_proc)
 implicit none
 interface
   real function dummy_proc(x)
    real :: x
   end function
 end interface
 call bar()
contains
! CHECK-LABEL: func private @_QFtest8Pbar(
! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.boxproc<() -> ()>>> {fir.host_assoc}) attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
subroutine bar()
  ! CHECK: %[[tupAddr:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.boxproc<() -> ()>>>, i32) -> !fir.ref<!fir.boxproc<() -> ()>>
  ! CHECK: %[[dummyProc:.*]] = fir.load %[[tupAddr]] : !fir.ref<!fir.boxproc<() -> ()>>
  ! CHECK: %[[dummyProcCast:.*]] = fir.box_addr %[[dummyProc]] : (!fir.boxproc<() -> ()>) -> ((!fir.ref<f32>) -> f32)
  ! CHECK: fir.call %[[dummyProcCast]](%{{.*}}) {{.*}}: (!fir.ref<f32>) -> f32
 print *, dummy_proc(42.)
end subroutine
end subroutine

! Test capture of dummy subroutines.
subroutine test9(dummy_proc)
 implicit none
 interface
   subroutine dummy_proc()
   end subroutine
 end interface
 call bar()
contains
! CHECK-LABEL: func private @_QFtest9Pbar(
! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.boxproc<() -> ()>>> {fir.host_assoc}) attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
subroutine bar()
  ! CHECK: %[[tupAddr:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.boxproc<() -> ()>>>, i32) -> !fir.ref<!fir.boxproc<() -> ()>>
  ! CHECK: %[[dummyProc:.*]] = fir.load %[[tupAddr]] : !fir.ref<!fir.boxproc<() -> ()>>
  ! CHECK: %[[pa:.*]] = fir.box_addr %[[dummyProc]]
  ! CHECK: fir.call %[[pa]]() {{.*}}: () -> ()
  call dummy_proc()
end subroutine
end subroutine

! Test capture of namelist
! CHECK-LABEL: func @_QPtest10(
! CHECK-SAME: %[[i:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>{{.*}}) {
subroutine test10(i)
 implicit none
 integer, pointer :: i(:)
 namelist /a_namelist/ i
 ! CHECK: %[[tupAddr:.*]] = fir.coordinate_of %[[tup:.*]], %c0{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>
 ! CHECK: fir.store %[[i]] to %[[tupAddr]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>
 ! CHECK: fir.call @_QFtest10Pbar(%[[tup]]) {{.*}}: (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>>) -> ()
 call bar()
contains
! CHECK-LABEL: func private @_QFtest10Pbar(
! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>> {fir.host_assoc}) attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
subroutine bar()
  ! CHECK: %[[tupAddr:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>
  ! CHECK: fir.load %[[tupAddr]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>
  read (88, NML = a_namelist) 
end subroutine
end subroutine

! Test passing an internal procedure as a dummy argument.

! CHECK-LABEL: func @_QPtest_proc_dummy() {
! CHECK:         %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_proc_dummyEi"}
! CHECK:         %[[VAL_5:.*]] = fir.alloca tuple<!fir.ref<i32>>
! CHECK:         %[[VAL_7:.*]] = fir.address_of(@_QFtest_proc_dummyPtest_proc_dummy_a) : (!fir.ref<i32>, !fir.ref<tuple<!fir.ref<i32>>>) -> ()
! CHECK:         %[[VAL_8:.*]] = fir.emboxproc %[[VAL_7]], %[[VAL_5]] : ((!fir.ref<i32>, !fir.ref<tuple<!fir.ref<i32>>>) -> (), !fir.ref<tuple<!fir.ref<i32>>>) -> !fir.boxproc<() -> ()>
! CHECK:         fir.call @_QPtest_proc_dummy_other(%[[VAL_8]]) {{.*}}: (!fir.boxproc<() -> ()>) -> ()

! CHECK-LABEL: func private @_QFtest_proc_dummyPtest_proc_dummy_a(
! CHECK-SAME:          %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "j"},
! CHECK-SAME:          %[[VAL_1:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:         %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:         %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_7:.*]] = arith.addi %[[VAL_5]], %[[VAL_6]] : i32
! CHECK:         fir.store %[[VAL_7]] to %[[VAL_4]] : !fir.ref<i32>
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func @_QPtest_proc_dummy_other(
! CHECK-SAME:           %[[VAL_0:.*]]: !fir.boxproc<() -> ()>) {
! CHECK:         %[[VAL_1:.*]] = arith.constant 4 : i32
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref}
! CHECK:         fir.store %[[VAL_1]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_3:.*]] = fir.box_addr %[[VAL_0]] : (!fir.boxproc<() -> ()>) -> ((!fir.ref<i32>) -> ())
! CHECK:         fir.call %[[VAL_3]](%[[VAL_2]]) {{.*}}: (!fir.ref<i32>) -> ()
! CHECK:         return
! CHECK:       }

subroutine test_proc_dummy
  integer i
  i = 1
  call test_proc_dummy_other(test_proc_dummy_a)
  print *, i
contains
  subroutine test_proc_dummy_a(j)
    i = i + j
  end subroutine test_proc_dummy_a
end subroutine test_proc_dummy

subroutine test_proc_dummy_other(proc)
  call proc(4)
end subroutine test_proc_dummy_other

! CHECK-LABEL: func @_QPtest_proc_dummy_char() {
! CHECK-DAG:         %[[VAL_0:.*]] = arith.constant 10 : index
! CHECK-DAG:         %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK-DAG:         %[[VAL_2:.*]] = arith.constant 9 : index
! CHECK-DAG:         %[[VAL_3:.*]] = arith.constant false
! CHECK-DAG:         %[[VAL_4:.*]] = arith.constant 1 : index
! CHECK-DAG:         %[[VAL_5:.*]] = arith.constant 32 : i8
! CHECK-DAG:         %[[VAL_6:.*]] = arith.constant 6 : i32
! CHECK-DAG:         %[[VAL_8:.*]] = arith.constant 10 : i64
! CHECK-DAG:         %[[VAL_9:.*]] = arith.constant 40 : index
! CHECK-DAG:         %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_11:.*]] = fir.alloca !fir.char<1,40> {bindc_name = ".result"}
! CHECK:         %[[VAL_12:.*]] = fir.alloca !fir.char<1,10> {bindc_name = "message", uniq_name = "_QFtest_proc_dummy_charEmessage"}
! CHECK:         %[[VAL_13:.*]] = fir.alloca tuple<!fir.boxchar<1>>
! CHECK:         %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_13]], %[[VAL_1]] : (!fir.ref<tuple<!fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
! CHECK:         %[[VAL_16:.*]] = fir.emboxchar %[[VAL_12]], %[[VAL_0]] : (!fir.ref<!fir.char<1,10>>, index) -> !fir.boxchar<1>
! CHECK:         fir.store %[[VAL_16]] to %[[VAL_14]] : !fir.ref<!fir.boxchar<1>>
! CHECK:         %[[VAL_17:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,9>>
! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_2]] : (index) -> i64
! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_20:.*]] = fir.convert %[[VAL_17]] : (!fir.ref<!fir.char<1,9>>) -> !fir.ref<i8>
! CHECK:         fir.call @llvm.memmove.p0.p0.i64(%[[VAL_19]], %[[VAL_20]], %[[VAL_18]], %[[VAL_3]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:         %[[VAL_21:.*]] = fir.undefined !fir.char<1>
! CHECK:         %[[VAL_22:.*]] = fir.insert_value %[[VAL_21]], %[[VAL_5]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:         br ^bb1(%[[VAL_2]], %[[VAL_4]] : index, index)
! CHECK:       ^bb1(%[[VAL_23:.*]]: index, %[[VAL_24:.*]]: index):
! CHECK:         %[[VAL_25:.*]] = arith.cmpi sgt, %[[VAL_24]], %[[VAL_10]] : index
! CHECK:         cond_br %[[VAL_25]], ^bb2, ^bb3
! CHECK:       ^bb2:
! CHECK:         %[[VAL_26:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.array<10x!fir.char<1>>>
! CHECK:         %[[VAL_27:.*]] = fir.coordinate_of %[[VAL_26]], %[[VAL_23]] : (!fir.ref<!fir.array<10x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:         fir.store %[[VAL_22]] to %[[VAL_27]] : !fir.ref<!fir.char<1>>
! CHECK:         %[[VAL_28:.*]] = arith.addi %[[VAL_23]], %[[VAL_4]] : index
! CHECK:         %[[VAL_29:.*]] = arith.subi %[[VAL_24]], %[[VAL_4]] : index
! CHECK:         br ^bb1(%[[VAL_28]], %[[VAL_29]] : index, index)
! CHECK:       ^bb3:
! CHECK:         %[[VAL_30:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,
! CHECK:         %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_32:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_6]], %[[VAL_31]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_33:.*]] = fir.address_of(@_QFtest_proc_dummy_charPgen_message) : (!fir.ref<!fir.char<1,10>>, index, !fir.ref<tuple<!fir.boxchar<1>>>) -> !fir.boxchar<1>
! CHECK:         %[[VAL_34:.*]] = fir.emboxproc %[[VAL_33]], %[[VAL_13]] : ((!fir.ref<!fir.char<1,10>>, index, !fir.ref<tuple<!fir.boxchar<1>>>) -> !fir.boxchar<1>, !fir.ref<tuple<!fir.boxchar<1>>>) -> !fir.boxproc<() -> ()>
! CHECK:         %[[VAL_35:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:         %[[VAL_36:.*]] = fir.insert_value %[[VAL_35]], %[[VAL_34]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:         %[[VAL_37:.*]] = fir.insert_value %[[VAL_36]], %[[VAL_8]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:         %[[VAL_38:.*]] = fir.call @llvm.stacksave.p0() {{.*}}: () -> !fir.ref<i8>
! CHECK:         %[[VAL_39:.*]] = fir.call @_QPget_message(%[[VAL_11]], %[[VAL_9]], %[[VAL_37]]) {{.*}}: (!fir.ref<!fir.char<1,40>>, index, tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxchar<1>
! CHECK:         %[[VAL_40:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<!fir.char<1,40>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_41:.*]] = fir.convert %[[VAL_9]] : (index) -> i64
! CHECK:         %[[VAL_42:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_32]], %[[VAL_40]], %[[VAL_41]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:         fir.call @llvm.stackrestore.p0(%[[VAL_38]]) {{.*}}: (!fir.ref<i8>) -> ()
! CHECK:         %[[VAL_43:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_32]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func private @_QFtest_proc_dummy_charPgen_message(
! CHECK-SAME:                                            %[[VAL_0:.*]]: !fir.ref<!fir.char<1,10>>,
! CHECK-SAME:                                            %[[VAL_1:.*]]: index,
! CHECK-SAME:                                            %[[VAL_2:.*]]: !fir.ref<tuple<!fir.boxchar<1>>> {fir.host_assoc}) -> !fir.boxchar<1> attributes {fir.internal_proc, llvm.linkage = #llvm.linkage<internal>} {
! CHECK-DAG:         %[[VAL_3:.*]] = arith.constant 0 : i32
! CHECK-DAG:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK-DAG:         %[[VAL_5:.*]] = arith.constant false
! CHECK-DAG:         %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK-DAG:         %[[VAL_7:.*]] = arith.constant 32 : i8
! CHECK-DAG:         %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_3]] : (!fir.ref<tuple<!fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
! CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_9]] : !fir.ref<!fir.boxchar<1>>
! CHECK:         %[[VAL_11:.*]]:2 = fir.unboxchar %[[VAL_10]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_11]]#1, %[[VAL_4]] : index
! CHECK:         %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_4]], %[[VAL_11]]#1 : index
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (index) -> i64
! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_11]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:         fir.call @llvm.memmove.p0.p0.i64(%[[VAL_16]], %[[VAL_17]], %[[VAL_15]], %[[VAL_5]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:         %[[VAL_18:.*]] = fir.undefined !fir.char<1>
! CHECK:         %[[VAL_19:.*]] = fir.insert_value %[[VAL_18]], %[[VAL_7]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:         %[[VAL_20:.*]] = arith.subi %[[VAL_4]], %[[VAL_14]] : index
! CHECK:         br ^bb1(%[[VAL_14]], %[[VAL_20]] : index, index)
! CHECK:       ^bb1(%[[VAL_21:.*]]: index, %[[VAL_22:.*]]: index):
! CHECK:         %[[VAL_23:.*]] = arith.cmpi sgt, %[[VAL_22]], %[[VAL_8]] : index
! CHECK:         cond_br %[[VAL_23]], ^bb2, ^bb3
! CHECK:       ^bb2:
! CHECK:         %[[VAL_24:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.array<10x!fir.char<1>>>
! CHECK:         %[[VAL_25:.*]] = fir.coordinate_of %[[VAL_24]], %[[VAL_21]] : (!fir.ref<!fir.array<10x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:         fir.store %[[VAL_19]] to %[[VAL_25]] : !fir.ref<!fir.char<1>>
! CHECK:         %[[VAL_26:.*]] = arith.addi %[[VAL_21]], %[[VAL_6]] : index
! CHECK:         %[[VAL_27:.*]] = arith.subi %[[VAL_22]], %[[VAL_6]] : index
! CHECK:         br ^bb1(%[[VAL_26]], %[[VAL_27]] : index, index)
! CHECK:       ^bb3:
! CHECK:         %[[VAL_28:.*]] = fir.emboxchar %[[VAL_0]], %[[VAL_4]] : (!fir.ref<!fir.char<1,10>>, index) -> !fir.boxchar<1>
! CHECK:         return %[[VAL_28]] : !fir.boxchar<1>
! CHECK:       }

! CHECK-LABEL: func @_QPget_message(
! CHECK-SAME:                       %[[VAL_0:.*]]: !fir.ref<!fir.char<1,40>>,
! CHECK-SAME:                       %[[VAL_1:.*]]: index,
! CHECK-SAME:                       %[[VAL_2:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc}) -> !fir.boxchar<1> {
! CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 40 : index
! CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 12 : index
! CHECK-DAG:     %[[VAL_5:.*]] = arith.constant false
! CHECK-DAG:     %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK-DAG:     %[[VAL_7:.*]] = arith.constant 32 : i8
! CHECK-DAG:     %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_10:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,12>>
! CHECK:         %[[VAL_11:.*]] = fir.extract_value %[[VAL_2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:         %[[VAL_12:.*]] = fir.box_addr %[[VAL_11]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:         %[[VAL_13:.*]] = fir.extract_value %[[VAL_2]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> i64
! CHECK:         %[[VAL_14:.*]] = fir.call @llvm.stacksave.p0() {{.*}}: () -> !fir.ref<i8>
! CHECK:         %[[VAL_15:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_13]] : i64) {bindc_name = ".result"}
! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_12]] : (() -> ()) -> ((!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>)
! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:         %[[VAL_18:.*]] = fir.call %[[VAL_16]](%[[VAL_15]], %[[VAL_17]]) {{.*}}: (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:         %[[VAL_19:.*]] = arith.addi %[[VAL_17]], %[[VAL_4]] : index
! CHECK:         %[[VAL_20:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_19]] : index) {bindc_name = ".chrtmp"}
! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_4]] : (index) -> i64
! CHECK:         %[[VAL_22:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_23:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.char<1,12>>) -> !fir.ref<i8>
! CHECK:         fir.call @llvm.memmove.p0.p0.i64(%[[VAL_22]], %[[VAL_23]], %[[VAL_21]], %[[VAL_5]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:         br ^bb1(%[[VAL_4]], %[[VAL_17]] : index, index)
! CHECK:       ^bb1(%[[VAL_24:.*]]: index, %[[VAL_25:.*]]: index):
! CHECK:         %[[VAL_26:.*]] = arith.cmpi sgt, %[[VAL_25]], %[[VAL_8]] : index
! CHECK:         cond_br %[[VAL_26]], ^bb2, ^bb3
! CHECK:       ^bb2:
! CHECK:         %[[VAL_27:.*]] = arith.subi %[[VAL_24]], %[[VAL_4]] : index
! CHECK:         %[[VAL_28:.*]] = fir.convert %[[VAL_15]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:         %[[VAL_29:.*]] = fir.coordinate_of %[[VAL_28]], %[[VAL_27]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:         %[[VAL_30:.*]] = fir.load %[[VAL_29]] : !fir.ref<!fir.char<1>>
! CHECK:         %[[VAL_31:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:         %[[VAL_32:.*]] = fir.coordinate_of %[[VAL_31]], %[[VAL_24]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:         fir.store %[[VAL_30]] to %[[VAL_32]] : !fir.ref<!fir.char<1>>
! CHECK:         %[[VAL_33:.*]] = arith.addi %[[VAL_24]], %[[VAL_6]] : index
! CHECK:         %[[VAL_34:.*]] = arith.subi %[[VAL_25]], %[[VAL_6]] : index
! CHECK:         br ^bb1(%[[VAL_33]], %[[VAL_34]] : index, index)
! CHECK:       ^bb3:
! CHECK:         %[[VAL_35:.*]] = arith.cmpi sgt, %[[VAL_19]], %[[VAL_3]] : index
! CHECK:         %[[VAL_36:.*]] = arith.select %[[VAL_35]], %[[VAL_3]], %[[VAL_19]] : index
! CHECK:         %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (index) -> i64
! CHECK:         %[[VAL_38:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.char<1,40>>) -> !fir.ref<i8>
! CHECK:         fir.call @llvm.memmove.p0.p0.i64(%[[VAL_38]], %[[VAL_22]], %[[VAL_37]], %[[VAL_5]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:         %[[VAL_39:.*]] = fir.undefined !fir.char<1>
! CHECK:         %[[VAL_40:.*]] = fir.insert_value %[[VAL_39]], %[[VAL_7]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:         %[[VAL_41:.*]] = arith.subi %[[VAL_3]], %[[VAL_36]] : index
! CHECK:         br ^bb4(%[[VAL_36]], %[[VAL_41]] : index, index)
! CHECK:       ^bb4(%[[VAL_42:.*]]: index, %[[VAL_43:.*]]: index):
! CHECK:         %[[VAL_44:.*]] = arith.cmpi sgt, %[[VAL_43]], %[[VAL_8]] : index
! CHECK:         cond_br %[[VAL_44]], ^bb5, ^bb6
! CHECK:       ^bb5:
! CHECK:         %[[VAL_45:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.char<1,40>>) -> !fir.ref<!fir.array<40x!fir.char<1>>>
! CHECK:         %[[VAL_46:.*]] = fir.coordinate_of %[[VAL_45]], %[[VAL_42]] : (!fir.ref<!fir.array<40x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:         fir.store %[[VAL_40]] to %[[VAL_46]] : !fir.ref<!fir.char<1>>
! CHECK:         %[[VAL_47:.*]] = arith.addi %[[VAL_42]], %[[VAL_6]] : index
! CHECK:         %[[VAL_48:.*]] = arith.subi %[[VAL_43]], %[[VAL_6]] : index
! CHECK:         br ^bb4(%[[VAL_47]], %[[VAL_48]] : index, index)
! CHECK:       ^bb6:
! CHECK:         fir.call @llvm.stackrestore.p0(%[[VAL_14]]) {{.*}}: (!fir.ref<i8>) -> ()
! CHECK:         %[[VAL_49:.*]] = fir.emboxchar %[[VAL_0]], %[[VAL_3]] : (!fir.ref<!fir.char<1,40>>, index) -> !fir.boxchar<1>
! CHECK:         return %[[VAL_49]] : !fir.boxchar<1>
! CHECK:       }

subroutine test_proc_dummy_char
  character(40) get_message
  external get_message
  character(10) message
  message = "Hi there!"
  print *, get_message(gen_message)
contains
  function gen_message
    character(10) :: gen_message
    gen_message = message
  end function gen_message
end subroutine test_proc_dummy_char

function get_message(a)
  character(40) :: get_message
  character(*) :: a
  get_message = "message is: " // a() 
end function get_message

! CHECK-LABEL: func @_QPtest_11a() {
! CHECK: %[[a:.*]] = fir.address_of(@_QPtest_11b) : () -> ()
! CHECK: %[[b:.*]] = fir.emboxproc %[[a]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: fir.call @_QPtest_11c(%[[b]], %{{.*}}) {{.*}}: (!fir.boxproc<() -> ()>, !fir.ref<i32>) -> ()
! CHECK: func private @_QPtest_11c(!fir.boxproc<() -> ()>, !fir.ref<i32>)

subroutine test_11a
  external test_11b
  call test_11c(test_11b, 3)
end subroutine test_11a

subroutine test_PDT_with_init_do_not_crash_host_symbol_analysis()
  integer :: i
  call sub()
contains
  subroutine sub()
    ! PDT definition symbols maps to un-analyzed expression,
    ! check this does not crash the visit of the internal procedure
    ! parse-tree to get the list of captured host variables.
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
