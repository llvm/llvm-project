! Test the names created for globals holding constant literal values
! RUN: bbc -emit-fir -o - %s | FileCheck %s

type someType
  integer :: i
end type

type otherType
  integer :: i
end type

type emptyType1
end type emptyType1

type emptyType2
end type emptyType2

  print *, [42, 42]
! CHECK: fir.address_of(@_QQro.2xi4.0)

  print *, reshape([42, 42, 42, 42, 42, 42], [2,3])
! CHECK: fir.address_of(@_QQro.2x3xi4.1)

  print *, [42_8, 42_8]
! CHECK: fir.address_of(@_QQro.2xi8.2)

  print *, [0.42, 0.42]
! CHECK: fir.address_of(@_QQro.2xr4.3)

  print *, [0.42_8, 0.42_8]
! CHECK: fir.address_of(@_QQro.2xr8.4)

  print *, [.true.]
! CHECK: fir.address_of(@_QQro.1xl4.5)

  print *, [.true._8]
! CHECK: fir.address_of(@_QQro.1xl8.6)

  print *, [(1., -1.), (-1., 1)]
! CHECK: fir.address_of(@_QQro.2xz4.7)

  print *, [(1._8, -1._8), (-1._8, 1._8)]
! CHECK: fir.address_of(@_QQro.2xz8.8)

  print *, [someType(42), someType(43)]
! CHECK: fir.address_of(@_QQro.2x_QFTsometype.9

  ! Verify that literals of the same type/shape
  ! are mapped to different global objects:
  print *, [someType(11)]
! CHECK: fir.address_of(@_QQro.1x_QFTsometype.10)
  print *, [someType(42)]
! CHECK: fir.address_of(@_QQro.1x_QFTsometype.11)
  print *, [someType(11)]
! CHECK: fir.address_of(@_QQro.1x_QFTsometype.10)
  print *, [someType(42)]
! CHECK: fir.address_of(@_QQro.1x_QFTsometype.11)
  print *, [someType(11)]
! CHECK: fir.address_of(@_QQro.1x_QFTsometype.10)
  print *, [someType(42)]
! CHECK: fir.address_of(@_QQro.1x_QFTsometype.11)
  print *, [someType(11)]
! CHECK: fir.address_of(@_QQro.1x_QFTsometype.10)
  print *, [someType(42)]
! CHECK: fir.address_of(@_QQro.1x_QFTsometype.11)

  print *, [Character(4)::]
! CHECK: fir.address_of(@_QQro.0x4xc1.null.12)
  print *, [Character(2)::]
! CHECK: fir.address_of(@_QQro.0x2xc1.null.13)
  print *, [Character(2)::]
! CHECK: fir.address_of(@_QQro.0x2xc1.null.13)

  print *, [otherType(42)]
! CHECK: fir.address_of(@_QQro.1x_QFTothertype.14)

  print *, [emptyType1()]
  print *, [emptyType2()]
end

! CHECK: fir.global internal @_QQro.1x_QFTsometype.10 constant : !fir.array<1x!fir.type<_QFTsometype{i:i32}>> {
! CHECK:   %{{.*}} = arith.constant 11 : i32
! CHECK: }

! CHECK: fir.global internal @_QQro.1x_QFTsometype.11 constant : !fir.array<1x!fir.type<_QFTsometype{i:i32}>> {
! CHECK:   %{{.*}} = arith.constant 42 : i32
! CHECK: }

! CHECK: fir.global internal @_QQro.0x4xc1.null.12 constant : !fir.array<0x!fir.char<1,4>> {
! CHECK:   %[[T1:.*]] = fir.undefined !fir.array<0x!fir.char<1,4>>
! CHECK:   fir.has_value %[[T1]] : !fir.array<0x!fir.char<1,4>>
! CHECK: }

! CHECK: fir.global internal @_QQro.0x2xc1.null.13 constant : !fir.array<0x!fir.char<1,2>> {
! CHECK:   %[[T2:.*]] = fir.undefined !fir.array<0x!fir.char<1,2>>
! CHECK:   fir.has_value %[[T2]] : !fir.array<0x!fir.char<1,2>>
! CHECK: }

! CHECK: fir.global internal @_QQro.1x_QFTothertype.14 constant : !fir.array<1x!fir.type<_QFTothertype{i:i32}>> {
! CHECK:   %{{.*}} = arith.constant 42 : i32
! CHECK: }
