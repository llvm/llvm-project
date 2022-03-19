! RUN: bbc -o - %s | FileCheck %s

! CHECK-LABEL: _QPzero1
subroutine zero1(z)
  real, dimension(:) :: z
  print*, size(z), z, ':'
end

! CHECK-LABEL: _QPzero2
subroutine zero2
  type dt
    integer :: j = 17
  end type
  ! CHECK: %[[z:[0-9]*]] = fir.alloca !fir.array<0x!fir.type<_QFzero2Tdt{j:i32}>> {bindc_name = "z", uniq_name = "_QFzero2Ez"}
  ! CHECK: %[[shape:[0-9]*]] = fir.shape %c0 : (index) -> !fir.shape<1>
  ! CHECK: fir.embox %[[z]](%[[shape]]) : (!fir.ref<!fir.array<0x!fir.type<_QFzero2Tdt{j:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<0x!fir.type<_QFzero2Tdt{j:i32}>>>
  type(dt) :: z(0)
  print*, size(z), z, ':'
end

! CHECK-LABEL: _QPzero3
subroutine zero3
  type dt
    integer :: j
  end type
  ! CHECK: %[[z:[0-9]*]] = fir.address_of(@_QFzero3Ez) : !fir.ref<!fir.array<0x!fir.type<_QFzero3Tdt{j:i32}>>>
  ! CHECK: %[[shape:[0-9]*]] = fir.shape %c0 : (index) -> !fir.shape<1>
  ! CHECK: fir.embox %[[z]](%[[shape]]) : (!fir.ref<!fir.array<0x!fir.type<_QFzero3Tdt{j:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<0x!fir.type<_QFzero3Tdt{j:i32}>>>
  type(dt) :: z(0) = dt(99)
  print*, size(z), z, ':'
end

! CHECK-LABEL: _QQmain
program prog
  real nada(2:-1)
  interface
    subroutine zero1(aa)
      real, dimension(:) :: aa
    end
  end interface
  ! CHECK: %[[shape:[0-9]*]] = fir.shape_shift %c2, %c0 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %2 = fir.embox %0(%[[shape]]) : (!fir.ref<!fir.array<0xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<0xf32>>
  call zero1(nada)
  call zero2
  call zero3
end
