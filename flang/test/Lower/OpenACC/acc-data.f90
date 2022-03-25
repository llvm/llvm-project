! This test checks lowering of OpenACC data directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

subroutine acc_data
  real, dimension(10, 10) :: a, b, c
  real, pointer :: d, e
  logical :: ifCondition = .TRUE.

!CHECK: [[A:%.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ea"}
!CHECK: [[B:%.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Eb"}
!CHECK: [[C:%.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ec"}
!CHECK: [[D:%.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "d", uniq_name = "{{.*}}Ed"}
!CHECK: [[E:%.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "e", uniq_name = "{{.*}}Ee"}

  !$acc data if(.TRUE.) copy(a)
  !$acc end data

!CHECK:      [[IF1:%.*]] = arith.constant true
!CHECK:      acc.data if([[IF1]]) copy([[A]] : !fir.ref<!fir.array<10x10xf32>>)  {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

  !$acc data copy(a) if(ifCondition)
  !$acc end data

!CHECK:      [[IFCOND:%.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
!CHECK:      [[IF2:%.*]] = fir.convert [[IFCOND]] : (!fir.logical<4>) -> i1
!CHECK:      acc.data if([[IF2]]) copy([[A]] : !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

  !$acc data copy(a, b, c)
  !$acc end data

!CHECK:      acc.data copy([[A]], [[B]], [[C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

  !$acc data copy(a) copy(b) copy(c)
  !$acc end data

!CHECK:      acc.data copy([[A]], [[B]], [[C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

  !$acc data copyin(a) copyin(readonly: b, c)
  !$acc end data

!CHECK:      acc.data copyin([[A]] : !fir.ref<!fir.array<10x10xf32>>) copyin_readonly([[B]], [[C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

  !$acc data copyout(a) copyout(zero: b) copyout(c)
  !$acc end data

!CHECK:      acc.data copyout([[A]], [[C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) copyout_zero([[B]] : !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

  !$acc data create(a, b) create(zero: c)
  !$acc end data

!CHECK:      acc.data create([[A]], [[B]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) create_zero([[C]] : !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

  !$acc data no_create(a, b) create(zero: c)
  !$acc end data

!CHECK:      acc.data create_zero([[C]] : !fir.ref<!fir.array<10x10xf32>>) no_create([[A]], [[B]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

  !$acc data present(a, b, c)
  !$acc end data

!CHECK:      acc.data present([[A]], [[B]], [[C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

  !$acc data deviceptr(b, c)
  !$acc end data

!CHECK:      acc.data deviceptr([[B]], [[C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

  !$acc data attach(d, e)
  !$acc end data

!CHECK:      acc.data attach([[D]], [[E]] : !fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

end subroutine acc_data

