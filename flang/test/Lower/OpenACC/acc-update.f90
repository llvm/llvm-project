! This test checks lowering of OpenACC update directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

subroutine acc_update
  integer :: async = 1
  real, dimension(10, 10) :: a, b, c
  logical :: ifCondition = .TRUE.

!CHECK: [[A:%.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ea"}
!CHECK: [[B:%.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Eb"}
!CHECK: [[C:%.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ec"}

  !$acc update host(a)
!CHECK: acc.update host([[A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc update host(a) if(.true.)
!CHECK: [[IF1:%.*]] = arith.constant true
!CHECK: acc.update if([[IF1]]) host([[A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc update host(a) if(ifCondition)
!CHECK: [[IFCOND:%.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
!CHECK: [[IF2:%.*]] = fir.convert [[IFCOND]] : (!fir.logical<4>) -> i1
!CHECK: acc.update if([[IF2]]) host([[A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc update host(a) host(b) host(c)
!CHECK: acc.update host([[A]], [[B]], [[C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc update host(a) host(b) device(c)
!CHECK: acc.update host([[A]], [[B]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) device([[C]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc update host(a) async
!CHECK: acc.update host([[A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {async}

  !$acc update host(a) wait
!CHECK: acc.update host([[A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {wait}

  !$acc update host(a) async wait
!CHECK: acc.update host([[A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {async, wait}

  !$acc update host(a) async(1)
!CHECK: [[ASYNC1:%.*]] = arith.constant 1 : i32
!CHECK: acc.update async([[ASYNC1]] : i32) host([[A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc update host(a) async(async)
!CHECK: [[ASYNC2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK: acc.update async([[ASYNC2]] : i32) host([[A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc update host(a) wait(1)
!CHECK: [[WAIT1:%.*]] = arith.constant 1 : i32
!CHECK: acc.update wait([[WAIT1]] : i32) host([[A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc update host(a) wait(queues: 1, 2)
!CHECK: [[WAIT2:%.*]] = arith.constant 1 : i32
!CHECK: [[WAIT3:%.*]] = arith.constant 2 : i32
!CHECK: acc.update wait([[WAIT2]], [[WAIT3]] : i32, i32) host([[A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc update host(a) wait(devnum: 1: queues: 1, 2)
!CHECK: [[WAIT4:%.*]] = arith.constant 1 : i32
!CHECK: [[WAIT5:%.*]] = arith.constant 2 : i32
!CHECK: [[WAIT6:%.*]] = arith.constant 1 : i32
!CHECK: acc.update wait_devnum([[WAIT6]] : i32) wait([[WAIT4]], [[WAIT5]] : i32, i32) host([[A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc update host(a) device_type(1, 2)
!CHECK: [[DEVTYPE1:%.*]] = arith.constant 1 : i32
!CHECK: [[DEVTYPE2:%.*]] = arith.constant 2 : i32
!CHECK: acc.update device_type([[DEVTYPE1]], [[DEVTYPE2]] : i32, i32) host([[A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc update host(a) device_type(*)
!CHECK: [[DEVTYPE3:%.*]] = arith.constant -1 : index
!CHECK: acc.update device_type([[DEVTYPE3]] : index) host([[A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}
end subroutine acc_update
