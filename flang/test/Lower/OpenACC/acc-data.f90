! This test checks lowering of OpenACC data directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

program acc_data
  real, dimension(10, 10) :: a, b, c

!CHECK: [[A:%.*]] = fir.alloca !fir.array<10x10xf32> {name = "a"}
!CHECK: [[B:%.*]] = fir.alloca !fir.array<10x10xf32> {name = "b"}
!CHECK: [[C:%.*]] = fir.alloca !fir.array<10x10xf32> {name = "c"}

  !$acc data copy(a, b, c)
  !$acc end data

!CHECK:      acc.data copy([[A]]: !fir.ref<!fir.array<10x10xf32>>, [[B]]: !fir.ref<!fir.array<10x10xf32>>, [[C]]: !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

  !$acc data copy(a) copy(b) copy(c)
  !$acc end data

!CHECK:      acc.data copy([[A]]: !fir.ref<!fir.array<10x10xf32>>, [[B]]: !fir.ref<!fir.array<10x10xf32>>, [[C]]: !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

  !$acc data copyin(a) copyin(readonly: b, c)
  !$acc end data

!CHECK:      acc.data copyin([[A]]: !fir.ref<!fir.array<10x10xf32>>) copyin_readonly([[B]]: !fir.ref<!fir.array<10x10xf32>>, [[C]]: !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

  !$acc data copyout(a) copyout(zero: b) copyout(c)
  !$acc end data

!CHECK:      acc.data copyout([[A]]: !fir.ref<!fir.array<10x10xf32>>, [[C]]: !fir.ref<!fir.array<10x10xf32>>) copyout_zero([[B]]: !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

  !$acc data create(a, b) create(zero: c)
  !$acc end data

!CHECK:      acc.data create([[A]]: !fir.ref<!fir.array<10x10xf32>>, [[B]]: !fir.ref<!fir.array<10x10xf32>>) create_zero([[C]]: !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

  !$acc data no_create(a, b) create(zero: c)
  !$acc end data

!CHECK:      acc.data create_zero([[C]]: !fir.ref<!fir.array<10x10xf32>>) no_create([[A]]: !fir.ref<!fir.array<10x10xf32>>, [[B]]: !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

  !$acc data present(a, b, c)
  !$acc end data

!CHECK:      acc.data present([[A]]: !fir.ref<!fir.array<10x10xf32>>, [[B]]: !fir.ref<!fir.array<10x10xf32>>, [[C]]: !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

  !$acc data attach(b, c)
  !$acc end data

!CHECK:      acc.data attach([[B]]: !fir.ref<!fir.array<10x10xf32>>, [[C]]: !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        acc.terminator
!CHECK-NEXT: }{{$}}

end program

