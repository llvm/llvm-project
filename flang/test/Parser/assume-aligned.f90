! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s

SUBROUTINE aa(a, nn)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nn
  COMPLEX(8), INTENT(INOUT), DIMENSION(1:nn) :: a
  INTEGER :: i
  !DIR$ assume_aligned a:16
!CHECK:  !DIR$ ASSUME_ALIGNED a:16
  !DIR$ assume_aligned a (1):16
!CHECK:  !DIR$ ASSUME_ALIGNED a(1):16  
  !DIR$ assume_aligned a(1):16
!CHECK:  !DIR$ ASSUME_ALIGNED a(1):16
  !DIR$ assume_aligned a(nn):16
!CHECK:  !DIR$ ASSUME_ALIGNED a(nn):16  
  !DIR$ assume_aligned a(44):16
!CHECK:  !DIR$ ASSUME_ALIGNED a(44):16  
  DO i=1,nn
     a(i)=a(i)+1.5
  END DO
END SUBROUTINE aa

SUBROUTINE bb(v, s, e)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: s(3), e(3)
  INTEGER :: y,z
  REAL(8),   INTENT(IN)  :: v(s(1):e(1),s(2):e(2),s(3):e(3))
  !DIR$ assume_aligned v(s(1),y,z)     :64
!CHECK: !DIR$ ASSUME_ALIGNED v(s(1),y,z):64
END SUBROUTINE bb

SUBROUTINE f(n)
  IMPLICIT NONE
  TYPE node 
    REAL(KIND=8), POINTER :: a(:,:)
  END TYPE NODE 
  
  TYPE(NODE), POINTER :: nodes
  INTEGER :: i
  INTEGER, INTENT(IN) :: n

  ALLOCATE(nodes) 
  ALLOCATE(nodes%a(1000,1000))

  !DIR$ ASSUME_ALIGNED nodes%a(1,1) : 16               
!CHECK: !DIR$ ASSUME_ALIGNED nodes%a(1,1):16
  DO i=1,n 
    nodes%a(1,i) = nodes%a(1,i)+1 
  END DO 
END SUBROUTINE f

SUBROUTINE g(a, b)
  IMPLICIT NONE
  INTEGER, INTENT(in) :: a(128), b(128)
  !DIR$ ASSUME_ALIGNED a:32, b:64
!CHECK: !DIR$ ASSUME_ALIGNED a:32, b:64
END SUBROUTINE g
