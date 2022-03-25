! This test checks lowering of OpenACC exit data directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

subroutine acc_exit_data
  integer :: async = 1
  real, dimension(10, 10) :: a, b, c
  real, pointer :: d
  logical :: ifCondition = .TRUE.

!CHECK: [[A:%.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ea"}
!CHECK: [[B:%.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Eb"}
!CHECK: [[C:%.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ec"}
!CHECK: [[D:%.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "d", uniq_name = "{{.*}}Ed"}

  !$acc exit data delete(a)
!CHECK: acc.exit_data delete([[A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc exit data delete(a) if(.true.)
!CHECK: [[IF1:%.*]] = arith.constant true
!CHECK: acc.exit_data if([[IF1]]) delete([[A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc exit data delete(a) if(ifCondition)
!CHECK: [[IFCOND:%.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
!CHECK: [[IF2:%.*]] = fir.convert [[IFCOND]] : (!fir.logical<4>) -> i1
!CHECK: acc.exit_data if([[IF2]]) delete([[A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc exit data delete(a) delete(b) delete(c)
!CHECK: acc.exit_data delete([[A]], [[B]], [[C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc exit data copyout(a) delete(b) detach(d)
!CHECK: acc.exit_data copyout([[A]] : !fir.ref<!fir.array<10x10xf32>>) delete([[B]] : !fir.ref<!fir.array<10x10xf32>>) detach([[D]] : !fir.ref<!fir.box<!fir.ptr<f32>>>){{$}}

  !$acc exit data delete(a) async
!CHECK: acc.exit_data delete([[A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {async}

  !$acc exit data delete(a) wait
!CHECK: acc.exit_data delete([[A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {wait}

  !$acc exit data delete(a) async wait
!CHECK: acc.exit_data delete([[A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {async, wait}

  !$acc exit data delete(a) async(1)
!CHECK: [[ASYNC1:%.*]] = arith.constant 1 : i32
!CHECK: acc.exit_data async([[ASYNC1]] : i32) delete([[A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc exit data delete(a) async(async)
!CHECK: [[ASYNC2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK: acc.exit_data async([[ASYNC2]] : i32) delete([[A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc exit data delete(a) wait(1)
!CHECK: [[WAIT1:%.*]] = arith.constant 1 : i32
!CHECK: acc.exit_data wait([[WAIT1]] : i32) delete([[A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc exit data delete(a) wait(queues: 1, 2)
!CHECK: [[WAIT2:%.*]] = arith.constant 1 : i32
!CHECK: [[WAIT3:%.*]] = arith.constant 2 : i32
!CHECK: acc.exit_data wait([[WAIT2]], [[WAIT3]] : i32, i32) delete([[A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc exit data delete(a) wait(devnum: 1: queues: 1, 2)
!CHECK: [[WAIT4:%.*]] = arith.constant 1 : i32
!CHECK: [[WAIT5:%.*]] = arith.constant 2 : i32
!CHECK: [[WAIT6:%.*]] = arith.constant 1 : i32
!CHECK: acc.exit_data wait_devnum([[WAIT6]] : i32) wait([[WAIT4]], [[WAIT5]] : i32, i32) delete([[A]] : !fir.ref<!fir.array<10x10xf32>>)

end subroutine acc_exit_data
