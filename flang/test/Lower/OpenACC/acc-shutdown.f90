! This test checks lowering of OpenACC shutdown directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

subroutine acc_shutdown
  logical :: ifCondition = .TRUE.

  !$acc shutdown
!CHECK: acc.shutdown{{$}}

  !$acc shutdown if(.true.)
!CHECK: [[IF1:%.*]] = arith.constant true
!CHECK: acc.shutdown if([[IF1]]){{$}}

  !$acc shutdown if(ifCondition)
!CHECK: [[IFCOND:%.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
!CHECK: [[IF2:%.*]] = fir.convert [[IFCOND]] : (!fir.logical<4>) -> i1
!CHECK: acc.shutdown if([[IF2]]){{$}}

  !$acc shutdown device_num(1)
!CHECK: [[DEVNUM:%.*]] = arith.constant 1 : i32
!CHECK: acc.shutdown device_num([[DEVNUM]] : i32){{$}}

  !$acc shutdown device_num(1) device_type(1, 2)
!CHECK: [[DEVNUM:%.*]] = arith.constant 1 : i32
!CHECK: [[DEVTYPE1:%.*]] = arith.constant 1 : i32
!CHECK: [[DEVTYPE2:%.*]] = arith.constant 2 : i32
!CHECK: acc.shutdown device_type([[DEVTYPE1]], [[DEVTYPE2]] : i32, i32) device_num([[DEVNUM]] : i32){{$}}

end subroutine acc_shutdown