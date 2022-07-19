! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of BGE, BGT, BLE, BLT

module testbge
  logical, parameter :: test_u = all((/&
       bge(0, 0), &
       bge(1, 1), &
       bge(2, 1), &
       bge(2147483647, 2147483647), &
       bge(2147483647, 2147483646), &
       bge(-1, -1), &
       bge(-1, -2), &
       bge(-2147483646, -2147483646), &
       bge(-2147483646, -2147483647), &
       bge(-1, 0), &
       bge(1, 0), &
       bge(-2147483647, 2147483647), &
       bge(Z'80000000', 2147483647)/))

  logical, parameter :: test_m = all((/&
       bge(1_4, 1_8), &
       bge(1_8, 1_4), &
       bge(-1_8, -1_4), &
       bge(-1_8, Z'FFFFFFFF'), &
       bge(Z'FFFFFFFFFFFFFFFF', -1_4)/))

  logical,parameter :: test_nm = all((/&
       .not. bge(-1_4, -1_8), &
       .not. bge(Z'FFFFFFFF', -1_8), &
       .not. bge(-1_4, Z'FFFFFFFFFFFFFFFF')/))
end module testbge

module testbgt
  logical, parameter :: test_u = all((/&
       bgt(2, 1), &
       bgt(2147483647, 2147483646), &
       bgt(-1, -2), &
       bgt(-2147483646, -2147483647), &
       bgt(-1, 0), &
       bgt(1, 0), &
       bgt(-2147483647, 2147483647), &
       bgt(Z'80000000', 2147483647) /))

  logical, parameter :: test_nu = all((/&
       .not. bgt(0, 0), &
       .not. bgt(1, 1), &
       .not. bgt(2147483647, 2147483647), &
       .not. bgt(-1, -1), &
       .not. bgt(-2147483646, -2147483646) /))

  logical, parameter :: test_m = all((/&
       bgt(-1_8, -1_4), &
       bgt(Z'FFFFFFFFFFFFFFFF', -1_4), &
       bgt(-1_8, Z'FFFFFFFF') /))

  logical, parameter :: test_nm = all((/&
       .not. bgt(1_4, 1_8), &
       .not. bgt(1_8, 1_4), &
       .not. bgt(-1_4, -1_8), &
       .not. bgt(Z'FFFFFFFF', -1_8), &
       .not. bgt(-1_4, Z'FFFFFFFFFFFFFFFF') /))
end module testbgt

module testble
  logical, parameter :: test_u = all((/&
       ble(0, 0), &
       ble(1, 1), &
       ble(1, 2), &
       ble(2147483647, 2147483647), &
       ble(2147483646, 2147483647), &
       ble(-1, -1), &
       ble(-2, -1), &
       ble(-2147483646, -2147483646), &
       ble(-2147483647, -2147483646), &
       ble(0, -1), &
       ble(0, 1), &
       ble(2147483647, -2147483647), &
       ble(2147483647, Z'80000000') /))

  logical, parameter :: test_m = all((/&
       ble(1_4, 1_8), &
       ble(1_8, 1_4), &
       ble(-1_4, -1_8), &
       ble(Z'FFFFFFFF', -1_8), &
       ble(-1_4, Z'FFFFFFFFFFFFFFFF') /))

  logical, parameter :: test_nm = all((/ &
       .not. ble(-1_8, -1_4), &
       .not. ble(Z'FFFFFFFFFFFFFFFF', -1_4), &
       .not. ble(-1_8, Z'FFFFFFFF') /))
end module testble

module testblt
  logical, parameter :: test_u = all((/&
       blt(1, 2), &
       blt(2147483646, 2147483647), &
       blt(-2, -1), &
       blt(-2147483647, -2147483646), &
       blt(0, -1), &
       blt(0, 1) /))

  logical, parameter :: test_nu = all((/&
       .not. blt(0, 0), &
       .not. blt(1, 1), &
       .not. blt(2147483647, 2147483647), &
       .not. blt(-1, -1), &
       .not. blt(-2147483646, -2147483646), &
       .not. blt(-2147483647, 2147483647), &
       .not. blt(Z'80000000', 2147483647)/))

  logical, parameter :: test_m = all((/&
       blt(-1_4, -1_8), &
       blt(Z'FFFFFFFF', -1_8), &
       blt(-1_4, Z'FFFFFFFFFFFFFFFF') /))

  logical, parameter :: test_nm = all ((/&
       .not. blt(1_4, 1_8), &
       .not. blt(1_8, 1_4), &
       .not. blt(-1_8, -1_4), &
       .not. blt(Z'FFFFFFFFFFFFFFFF', -1_4), &
       .not. blt(-1_8, Z'FFFFFFFF') /))
end module testblt
