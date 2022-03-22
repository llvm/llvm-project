! RUN: %python %S/test_errors.py %s %flang_fc1
! Confirm enforcement of constraints and restrictions in 7.7
! C7107, C7108, C7109

subroutine bozchecks
  ! Type declaration statements
  integer :: f, realpart = B"0101", img = B"1111", resint
  logical :: resbit
  complex :: rescmplx
  real :: dbl, e
  interface
    subroutine explicit(n, x, c)
      integer :: n
      real :: x
      character :: c
    end subroutine
  end interface
  ! C7107
  !ERROR: Invalid digit ('a') in BOZ literal 'b"110a"'
  integer, parameter :: a = B"110A"
  !ERROR: Invalid digit ('2') in BOZ literal 'b"1232"'
  integer, parameter :: b = B"1232"
  !ERROR: BOZ literal 'b"010101010101010101010101011111111111111111111111111111111111111111111111111111111111111111111111111111111111000000000000000000000000000000000000"' too large
  integer, parameter :: b1 = B"010101010101010101010101011111111111111111111&
                              &111111111111111111111111111111111111111111111&
                              &111111111111111111000000000000000000000000000&
                              &000000000"
  ! C7108
  !ERROR: Invalid digit ('8') in BOZ literal 'o"8"'
  integer :: c = O"8"
  !ERROR: Invalid digit ('a') in BOZ literal 'o"a"'
  integer :: d = O"A"

  ! C7109
  !    A) can appear only in data statement
  !    B) Argument to intrinsics listed from 16.9 below
  !       BGE, BGT, BLE, BLT, CMPLX, DBLE, DSHIFTL,
  !       DSHIFTR, IAND, IEOR, INT, IOR, MERGE_BITS, REAL
  !       and legacy aliases AND, OR, XOR

  ! part A
  data f / Z"AA" / ! OK
  !ERROR: DATA statement value could not be converted to the type 'COMPLEX(4)' of the object 'rescmplx'
  data rescmplx / B"010101" /
  ! part B
  resbit = BGE(B"0101", B"1111")
  resbit = BGT(Z"0101", B"1111")
  resbit = BLE(B"0101", B"1111")
  resbit = BLT(B"0101", B"1111")

  res = CMPLX (realpart, img, 4)
  res = CMPLX (B"0101", B"1111", 4)

  dbl = DBLE(B"1111")
  dbl = DBLE(realpart)

  !ERROR: Typeless (BOZ) not allowed for both 'i=' & 'j=' arguments
  dbl = DSHIFTL(B"0101",B"0101",2)
  !ERROR: Typeless (BOZ) not allowed for both 'i=' & 'j=' arguments
  dbl = DSHIFTR(B"1010",B"1010",2)
  dbl = DSHIFTL(B"0101",5,2) ! OK
  dbl = DSHIFTR(B"1010",5,2) ! OK

  !ERROR: Typeless (BOZ) not allowed for both 'i=' & 'j=' arguments
  resint = IAND(B"0001", B"0011")
  resint = IAND(B"0001", 3)
  !ERROR: Typeless (BOZ) not allowed for both 'i=' & 'j=' arguments
  resint = AND(B"0001", B"0011")
  resint = AND(B"0001", 3)

  !ERROR: Typeless (BOZ) not allowed for both 'i=' & 'j=' arguments
  resint = IEOR(B"0001", B"0011")
  resint = IEOR(B"0001", 3)
  !ERROR: Typeless (BOZ) not allowed for both 'i=' & 'j=' arguments
  resint = XOR(B"0001", B"0011")
  resint = XOR(B"0001", 3)

  resint = INT(B"1010")

  !ERROR: Typeless (BOZ) not allowed for both 'i=' & 'j=' arguments
  res = IOR(B"0101", B"0011")
  res = IOR(B"0101", 3)
  !ERROR: Typeless (BOZ) not allowed for both 'i=' & 'j=' arguments
  res = OR(B"0101", B"0011")
  res = OR(B"0101", 3)

  res = MERGE_BITS(13,3,11)
  res = MERGE_BITS(B"1101",3,11)
  !ERROR: Typeless (BOZ) not allowed for both 'i=' & 'j=' arguments
  res = MERGE_BITS(B"1101",B"0011",11)
  !ERROR: Typeless (BOZ) not allowed for both 'i=' & 'j=' arguments
  res = MERGE_BITS(B"1101",B"0011",B"1011")
  res = MERGE_BITS(B"1101",3,B"1011")

  !ERROR: Typeless (BOZ) not allowed for 'x=' argument
  res = KIND(z'feedface')

  res = REAL(B"1101")

  !Ok
  call explicit(z'deadbeef', o'666', 'a')

  !ERROR: Actual argument 'z'55'' associated with dummy argument 'c=' is not a variable or typed expression
  call explicit(z'deadbeef', o'666', b'01010101')

  !ERROR: BOZ argument requires an explicit interface
  call implictSub(Z'12345')

  !ERROR: Output item must not be a BOZ literal constant
  print "(Z18)", Z"76543210"
end subroutine
