! This test checks lowering of OpenACC wait directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine acc_update
  integer :: async = 1
  logical :: ifCondition = .TRUE.

  !$acc wait
!CHECK: acc.wait{{ *}}{{$}}

  !$acc wait if(.true.)
!CHECK: [[IF1:%.*]] = arith.constant true
!CHECK: acc.wait if([[IF1]]){{$}}

  !$acc wait if(ifCondition)
!CHECK: [[IFCOND:%.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
!CHECK: [[IF2:%.*]] = fir.convert [[IFCOND]] : (!fir.logical<4>) -> i1
!CHECK: acc.wait if([[IF2]]){{$}}

  !$acc wait(1, 2)
!CHECK: [[WAIT1:%.*]] = arith.constant 1 : i32
!CHECK: [[WAIT2:%.*]] = arith.constant 2 : i32
!CHECK: acc.wait([[WAIT1]], [[WAIT2]] : i32, i32){{ *}}{{$}}

  !$acc wait(1) async
!CHECK: [[WAIT3:%.*]] = arith.constant 1 : i32
!CHECK: acc.wait([[WAIT3]] : i32) attributes {async}

  !$acc wait(1) async(async)
!CHECK: [[WAIT3:%.*]] = arith.constant 1 : i32
!CHECK: [[ASYNC1:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK: acc.wait([[WAIT3]] : i32) async([[ASYNC1]] : i32){{$}}

  !$acc wait(devnum: 3: queues: 1, 2)
!CHECK: [[WAIT1:%.*]] = arith.constant 1 : i32
!CHECK: [[WAIT2:%.*]] = arith.constant 2 : i32
!CHECK: [[DEVNUM:%.*]] = arith.constant 3 : i32
!CHECK: acc.wait([[WAIT1]], [[WAIT2]] : i32, i32) wait_devnum([[DEVNUM]] : i32){{$}}

end subroutine acc_update
