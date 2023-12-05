! This test checks lowering of OpenACC init directive.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine acc_init
  implicit none
  logical :: ifCondition = .TRUE.
  integer :: ifInt = 1

  !$acc init
!CHECK: acc.init{{ *}}{{$}}

  !$acc init if(.true.)
!CHECK: [[IF1:%.*]] = arith.constant true
!CHECK: acc.init if([[IF1]]){{$}}

  !$acc init if(ifCondition)
!CHECK: [[IFCOND:%.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
!CHECK: [[IF2:%.*]] = fir.convert [[IFCOND]] : (!fir.logical<4>) -> i1
!CHECK: acc.init if([[IF2]]){{$}}

  !$acc init device_num(1)
!CHECK: [[DEVNUM:%.*]] = arith.constant 1 : i32
!CHECK: acc.init device_num([[DEVNUM]] : i32){{$}}

  !$acc init device_num(1) device_type(host, multicore)
!CHECK: [[DEVNUM:%.*]] = arith.constant 1 : i32
!CHECK: acc.init device_num([[DEVNUM]] : i32) attributes {device_types = [#acc.device_type<host>, #acc.device_type<multicore>]}

  !$acc init if(ifInt)
!CHECK: %[[IFINT:.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK: %[[CONV:.*]] = fir.convert %[[IFINT]] : (i32) -> i1
!CHECK: acc.init if(%[[CONV]])

   !$acc init device_type(nvidia)
!CHECK: acc.init attributes {device_types = [#acc.device_type<nvidia>]}

end subroutine acc_init
