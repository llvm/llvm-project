! RUN: %flang_fc1 -fdebug-unparse  %s  2>&1 | FileCheck %s
! Sometimes associations with named constants involving non-default
! lower bounds expose those bounds to LBOUND()/UBOUND(), sometimes
! they do not.
subroutine s(n)
  integer, intent(in) :: n
  type t
    real component(0:1,2:3)
  end type
  real, parameter :: abcd(2,2) = reshape([1.,2.,3.,4.], shape(abcd))
  real, parameter :: namedConst1(-1:0,-2:-1) = abcd
  type(t), parameter :: namedConst2 = t(abcd)
  type(t), parameter :: namedConst3(2:3,3:4) = reshape([(namedConst2,j=1,size(namedConst3))], shape(namedConst3))
!CHECK: PRINT *, [INTEGER(4)::1_4,1_4], [INTEGER(4)::2_4,2_4], [INTEGER(4)::2_4,2_4]
  print *, lbound(abcd), ubound(abcd), shape(abcd)
!CHECK: PRINT *, [INTEGER(4)::-1_4,-2_4], [INTEGER(4)::0_4,-1_4], [INTEGER(4)::2_4,2_4]
  print *, lbound(namedConst1), ubound(namedConst1), shape(namedConst1)
!CHECK: PRINT *, [INTEGER(4)::0_4,2_4], [INTEGER(4)::1_4,3_4], [INTEGER(4)::2_4,2_4]
  print *, lbound(namedConst2%component), ubound(namedConst2%component), shape(namedConst2%component)
!CHECK: PRINT *, [INTEGER(4)::2_4,3_4], [INTEGER(4)::3_4,4_4], [INTEGER(4)::2_4,2_4]
  print *, lbound(namedConst3), ubound(namedConst3), shape(namedConst3)
!CHECK: PRINT *, [INTEGER(4)::0_4,2_4], [INTEGER(4)::1_4,3_4], [INTEGER(4)::2_4,2_4]
  print *, lbound(namedConst3(n,n)%component), ubound(namedConst3(n,n)%component), shape(namedConst3(n,n)%component)
!CHECK: PRINT *, [INTEGER(4)::1_4,1_4], [INTEGER(4)::2_4,2_4], [INTEGER(4)::2_4,2_4]
  print *, lbound(namedConst3%component(0,2)), ubound(namedConst3%component(0,2)), shape(namedConst3%component(0,2))
  associate (a => abcd)
!CHECK:  PRINT *, [INTEGER(4)::1_4,1_4], [INTEGER(4)::2_4,2_4], [INTEGER(4)::2_4,2_4]
    print *, lbound(a), ubound(a), shape(a)
  end associate
  associate (a => namedConst1)
!CHECK:  PRINT *, [INTEGER(4)::-1_4,-2_4], [INTEGER(4)::0_4,-1_4], [INTEGER(4)::2_4,2_4]
    print *, lbound(a), ubound(a), shape(a)
  end associate
  associate (a => (namedConst1))
!CHECK:  PRINT *, [INTEGER(4)::1_4,1_4], [INTEGER(4)::2_4,2_4], [INTEGER(4)::2_4,2_4]
    print *, lbound(a), ubound(a), shape(a)
  end associate
  associate (a => namedConst1 * 2.)
!CHECK:  PRINT *, [INTEGER(4)::1_4,1_4], [INTEGER(4)::2_4,2_4], [INTEGER(4)::2_4,2_4]
    print *, lbound(a), ubound(a), shape(a)
  end associate
  associate (a => namedConst2%component)
!CHECK:  PRINT *, [INTEGER(4)::0_4,2_4], [INTEGER(4)::1_4,3_4], [INTEGER(4)::2_4,2_4]
    print *, lbound(a), ubound(a), shape(a)
  end associate
  associate (a => (namedConst2%component))
!CHECK:  PRINT *, [INTEGER(4)::1_4,1_4], [INTEGER(4)::2_4,2_4], [INTEGER(4)::2_4,2_4]
    print *, lbound(a), ubound(a), shape(a)
  end associate
  associate (a => namedConst2%component * 2.)
!CHECK:  PRINT *, [INTEGER(4)::1_4,1_4], [INTEGER(4)::2_4,2_4], [INTEGER(4)::2_4,2_4]
    print *, lbound(a), ubound(a), shape(a)
  end associate
  associate (a => namedConst3)
!CHECK:  PRINT *, [INTEGER(4)::2_4,3_4], [INTEGER(4)::3_4,4_4], [INTEGER(4)::2_4,2_4]
    print *, lbound(a), ubound(a), shape(a)
  end associate
  associate (a => (namedConst3))
!CHECK:  PRINT *, [INTEGER(4)::1_4,1_4], [INTEGER(4)::2_4,2_4], [INTEGER(4)::2_4,2_4]
    print *, lbound(a), ubound(a), shape(a)
  end associate
  associate (a => namedConst3(n,n)%component)
!CHECK:  PRINT *, [INTEGER(4)::0_4,2_4], [INTEGER(4)::1_4,3_4], [INTEGER(4)::2_4,2_4]
    print *, lbound(a), ubound(a), shape(a)
  end associate
  associate (a => (namedConst3(n,n)%component))
!CHECK:  PRINT *, [INTEGER(4)::1_4,1_4], [INTEGER(4)::2_4,2_4], [INTEGER(4)::2_4,2_4]
    print *, lbound(a), ubound(a), shape(a)
  end associate
  associate (a => namedConst3(n,n)%component * 2.)
!CHECK:  PRINT *, [INTEGER(4)::1_4,1_4], [INTEGER(4)::2_4,2_4], [INTEGER(4)::2_4,2_4]
    print *, lbound(a), ubound(a), shape(a)
  end associate
  associate (a => namedConst3%component(0,2))
!CHECK: PRINT *, [INTEGER(4)::1_4,1_4], [INTEGER(4)::2_4,2_4], [INTEGER(4)::2_4,2_4]
    print *, lbound(a), ubound(a), shape(a)
  end associate
end
