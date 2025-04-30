PROGRAM stress_test_nested_implied_do
  ! Test that a nested implied do loop can be used to initialize an array.
  ! Array constructors, and implicit DO loops are detailed in sec. 7.8
  ! of the Standard.
  IMPLICIT NONE
  INTEGER, PARAMETER :: n=4
  INTEGER, PARAMETER :: m=10
  INTEGER, PARAMETER :: o=2
  INTEGER, PARAMETER :: p=3
  INTEGER :: i, j, k, l

  ! 1 level of nesting
  INTEGER :: array1(n) = (/ (i, i=1, n) /)
  INTEGER :: expect1(n)
  data expect1/1, 2, 3, 4/

  ! 2 levels of nesting
  INTEGER :: array2(m) = (/ ((i, j=1, i), i=1, n ) /)
  INTEGER :: expect2(m)
  data expect2/1, 2, 2, 3, 3, 3, 4, 4, 4, 4/

  ! 3 levels of nesting
  INTEGER :: array3(m*o) = (/ (((i, j=1, i), i=1,n ), k=1, o) /)
  INTEGER :: expect3(m*o)
  data expect3/1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4/

  ! 4 levels of nesting
  INTEGER :: array4(m*o*p) = (/ ((((i, j=1, i), i=1,n ), k=1, o), l=1, p) /)
  INTEGER :: expect4(m*o*p)
  data expect4/1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4,&
               1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4,&
               1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4/

  ! 3 levels of nesting and multiple items
  INTEGER :: array5(3*m*o) = (/ (((i, i+10, i+100, j=1, i), i=1,n ), k=1, o) /)
  INTEGER :: expect5(3*m*o)
  data expect5/1, 11, 101,&
               2, 12, 102, 2, 12, 102,&
               3, 13, 103, 3, 13, 103, 3, 13, 103,&
               4, 14, 104, 4, 14, 104, 4, 14, 104, 4, 14, 104,&
               1, 11, 101,&
               2, 12, 102, 2, 12, 102,&
               3, 13, 103, 3, 13, 103, 3, 13, 103,&
               4, 14, 104, 4, 14, 104, 4, 14, 104, 4, 14, 104/

  INTEGER(KIND=1) :: i8, j8   ! TY_BINT
  INTEGER(KIND=2) :: i16, j16 ! TY_SINT
  INTEGER(KIND=4) :: i32, j32 ! TY_INT
  INTEGER(KIND=8) :: i64, j64 ! TY_INT8

  ! Different integers fitting into each other
  ! Array contents are intentionally oversized to show that a warning rather than an error is thrown

  INTEGER(KIND=1) :: array_i1_i1(6) = (/ ((256_1, j8=1_1, i8), i8=1_1, 3_1 ) /)
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 61}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 61}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 61}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 61}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 61}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 61}
  INTEGER(KIND=1) :: array_i1_i2(6) = (/ ((257_2, j16=1_2, i16), i16=1_2, 3_2 ) /)
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 68}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 68}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 68}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 68}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 68}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 68}
  INTEGER(KIND=1) :: array_i1_i4(6) = (/ ((257_4, j32=1_4, i32), i32=1_4, 3_4 ) /)
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 75}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 75}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 75}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 75}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 75}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 75}
  INTEGER(KIND=1) :: array_i1_i8(6) = (/ ((257_8, j64=1_8, i64), i64=1_8, 3_8 ) /)

  INTEGER(KIND=2) :: array_i2_i1(6) = (/ ((256_1, j8=1_1, i8), i8=1_1, 3_1 ) /)
  INTEGER(KIND=2) :: array_i2_i2(6) = (/ ((65536_2, j16=1_2, i16), i16=1_2, 3_2 ) /)
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 85}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 85}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 85}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 85}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 85}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 85}
  INTEGER(KIND=2) :: array_i2_i4(6) = (/ ((65537_4, j32=1_4, i32), i32=1_4, 3_4 ) /)
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 92}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 92}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 92}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 92}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 92}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 92}
  INTEGER(KIND=2) :: array_i2_i8(6) = (/ ((65537_8, j64=1_8, i64), i64=1_8, 3_8 ) /)

  INTEGER(KIND=4) :: array_i4_i1(6) = (/ ((256_1, j8=1_1, i8), i8=1_1, 3_1 ) /)
  INTEGER(KIND=4) :: array_i4_i2(6) = (/ ((65536_2, j16=1_2, i16), i16=1_2, 3_2 ) /)
  INTEGER(KIND=4) :: array_i4_i4(6) = (/ ((4294967296_4, j32=1_4, i32), i32=1_4, 3_4 ) /)
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 103}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 103}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 103}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 103}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 103}
  !{warning "PGF90-W-0128-Integer constant truncated to fit data type: 1" 103}
  INTEGER(KIND=4) :: array_i4_i8(6) = (/ ((4294967297_8, j64=1_8, i64), i64=1_8, 3_8 ) /)

  INTEGER(KIND=8) :: array_i8_i1(6) = (/ ((256_1, j8=1_1, i8), i8=1_1, 3_1 ) /)
  INTEGER(KIND=8) :: array_i8_i2(6) = (/ ((65536_2, j16=1_2, i16), i16=1_2, 3_2 ) /)
  INTEGER(KIND=8) :: array_i8_i4(6) = (/ ((4294967296_4, j32=1_4, i32), i32=1_4, 3_4 ) /)
  INTEGER(KIND=8) :: array_i8_i8(6) = (/ ((4294967297_8, j64=1_8, i64), i64=1_8, 3_8 ) /)

  ! Integers initializing real arrays

  REAL(KIND=4) :: array_r4_i1(6) = (/ ((256_1, j8=1_1, i8), i8=1_1, 3_1 ) /)
  REAL(KIND=4) :: array_r4_i2(6) = (/ ((65536_2, j16=1_2, i16), i16=1_2, 3_2 ) /)
  REAL(KIND=4) :: array_r4_i4(6) = (/ ((4294967296_4, j32=1_4, i32), i32=1_4, 3_4 ) /)
  REAL(KIND=4) :: array_r4_i8(6) = (/ ((4294967297_8, j64=1_8, i64), i64=1_8, 3_8 ) /)

  REAL(KIND=8) :: array_r8_i1(6) = (/ ((256_1, j8=1_1, i8), i8=1_1, 3_1 ) /)
  REAL(KIND=8) :: array_r8_i2(6) = (/ ((65536_2, j16=1_2, i16), i16=1_2, 3_2 ) /)
  REAL(KIND=8) :: array_r8_i4(6) = (/ ((4294967296_4, j32=1_4, i32), i32=1_4, 3_4 ) /)
  REAL(KIND=8) :: array_r8_i8(6) = (/ ((4294967297_8, j64=1_8, i64), i64=1_8, 3_8 ) /)

  call check(array1, expect1, n)
  call check(array2, expect2, m)
  call check(array3, expect3, m*o)
  call check(array4, expect4, m*o*p)
  call check(array5, expect5, 3*m*o)

END PROGRAM stress_test_nested_implied_do
