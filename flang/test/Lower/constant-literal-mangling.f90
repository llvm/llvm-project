! Test the names created for globals holding constant literal values
! RUN: bbc -emit-fir -o - %s | FileCheck %s

type someType
  integer :: i
end type

  print *, [42, 42]
! CHECK: fir.address_of(@_QQro.2xi4.53fa91e04725d4ee6f22cf1e2d38428a)

  print *, reshape([42, 42, 42, 42, 42, 42], [2,3])
! CHECK: fir.address_of(@_QQro.2x3xi4.9af8c8182bab45c4e7888ec3623db3b6)

  print *, [42_8, 42_8]
! CHECK: fir.address_of(@_QQro.2xi8.3b1356831516d19b976038974b2673ac)

  print *, [0.42, 0.42]
! CHECK: fir.address_of(@_QQro.2xr4.3c5becae2e4426ad1615e253139ceff8)

  print *, [0.42_8, 0.42_8]
! CHECK: fir.address_of(@_QQro.2xr8.ebefec8f7537fbf54acc4530e75084e6)

  print *, [.true.]
! CHECK: fir.address_of(@_QQro.1xl4.4352d88a78aa39750bf70cd6f27bcaa5)

  print *, [.true._8]
! CHECK: fir.address_of(@_QQro.1xl8.33cdeccccebe80329f1fdbee7f5874cb)

  print *, [(1., -1.), (-1., 1)]
! CHECK: fir.address_of(@_QQro.2xz4.ac09ecb1abceb4f9cad4b1a50000074e)

  print *, [(1._8, -1._8), (-1._8, 1._8)]
! CHECK: fir.address_of(@_QQro.2xz8.a3652db37055e37d2cae8198ae4cd959)

  print *, [someType(42), someType(43)]
! CHECK: fir.address_of(@_QQro.2x_QFTsometype.
! Note: the hash for derived types cannot clash with other constant in the same
! compilation unit, but is unstable because it hashes some noise contained in
! unused std::vector storage.
end
