! This test checks lowering of OpenACC init directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

subroutine acc_init
  logical :: ifCondition = .TRUE.

  !$acc init
!CHECK: acc.init{{$}}

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

  !$acc init device_num(1) device_type(1, 2)
!CHECK: [[DEVNUM:%.*]] = arith.constant 1 : i32
!CHECK: [[DEVTYPE1:%.*]] = arith.constant 1 : i32
!CHECK: [[DEVTYPE2:%.*]] = arith.constant 2 : i32
!CHECK: acc.init device_type([[DEVTYPE1]], [[DEVTYPE2]] : i32, i32) device_num([[DEVNUM]] : i32){{$}}

end subroutine acc_init