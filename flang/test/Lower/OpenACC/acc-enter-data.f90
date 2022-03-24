! This test checks lowering of OpenACC enter data directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

subroutine acc_enter_data
  integer :: async = 1
  real, dimension(10, 10) :: a, b, c
  real, pointer :: d
  logical :: ifCondition = .TRUE.

!CHECK: [[A:%.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ea"}
!CHECK: [[B:%.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Eb"}
!CHECK: [[C:%.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ec"}
!CHECK: [[D:%.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "d", uniq_name = "{{.*}}Ed"}

  !$acc enter data create(a)
!CHECK: acc.enter_data create([[A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc enter data create(a) if(.true.)
!CHECK: [[IF1:%.*]] = arith.constant true
!CHECK: acc.enter_data if([[IF1]]) create([[A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc enter data create(a) if(ifCondition)
!CHECK: [[IFCOND:%.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
!CHECK: [[IF2:%.*]] = fir.convert [[IFCOND]] : (!fir.logical<4>) -> i1
!CHECK: acc.enter_data if([[IF2]]) create([[A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc enter data create(a) create(b) create(c)
!CHECK: acc.enter_data create([[A]], [[B]], [[C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc enter data create(a) create(b) create(zero: c)
!CHECK: acc.enter_data create([[A]], [[B]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) create_zero([[C]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc enter data copyin(a) create(b) attach(d)
!CHECK: acc.enter_data copyin([[A]] : !fir.ref<!fir.array<10x10xf32>>) create([[B]] : !fir.ref<!fir.array<10x10xf32>>) attach([[D]] : !fir.ref<!fir.box<!fir.ptr<f32>>>){{$}}

  !$acc enter data create(a) async
!CHECK: acc.enter_data create([[A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {async}

  !$acc enter data create(a) wait
!CHECK: acc.enter_data create([[A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {wait}

  !$acc enter data create(a) async wait
!CHECK: acc.enter_data create([[A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {async, wait}

  !$acc enter data create(a) async(1)
!CHECK: [[ASYNC1:%.*]] = arith.constant 1 : i32
!CHECK: acc.enter_data async([[ASYNC1]] : i32) create([[A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data create(a) async(async)
!CHECK: [[ASYNC2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK: acc.enter_data async([[ASYNC2]] : i32) create([[A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data create(a) wait(1)
!CHECK: [[WAIT1:%.*]] = arith.constant 1 : i32
!CHECK: acc.enter_data wait([[WAIT1]] : i32) create([[A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data create(a) wait(queues: 1, 2)
!CHECK: [[WAIT2:%.*]] = arith.constant 1 : i32
!CHECK: [[WAIT3:%.*]] = arith.constant 2 : i32
!CHECK: acc.enter_data wait([[WAIT2]], [[WAIT3]] : i32, i32) create([[A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data create(a) wait(devnum: 1: queues: 1, 2)
!CHECK: [[WAIT4:%.*]] = arith.constant 1 : i32
!CHECK: [[WAIT5:%.*]] = arith.constant 2 : i32
!CHECK: [[WAIT6:%.*]] = arith.constant 1 : i32
!CHECK: acc.enter_data wait_devnum([[WAIT6]] : i32) wait([[WAIT4]], [[WAIT5]] : i32, i32) create([[A]] : !fir.ref<!fir.array<10x10xf32>>)

end subroutine acc_enter_data
