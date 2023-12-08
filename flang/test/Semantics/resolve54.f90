! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests based on examples in C.10.6

! C.10.6(10)
module m1
  interface GOOD1
    function F1A(X)
      real :: F1A, X
    end function
    function F1B(X)
      integer :: F1B, X
    end function
  end interface
end

! C.10.6(13)
module m2
  interface GOOD2
    function F2A(X)
      real :: F2A, X
    end function
    function F2B(X, Y)
      complex :: F2B
      real :: X, Y
    end function
  end interface
end

! C.10.6(15)
module m3
  interface GOOD3
    subroutine S3A(W, X, Y, Z)
      real :: W, Y
      integer :: X, Z
    end subroutine
    subroutine S3B(X, W, Z, Y)
      real :: W, Z
      integer :: X, Y
    end subroutine
  end interface
end
module m3b
  interface GOOD3
    subroutine S3B(X, W, Z, Y)
      real :: W, Z
      integer :: X, Y
    end subroutine
    subroutine S3A(W, X, Y, Z)
      real :: W, Y
      integer :: X, Z
    end subroutine
  end interface
end

! C.10.6(17)
! BAD4(1.0,2,Y=3.0,Z=4) could apply to either procedure
module m4
  !ERROR: Generic 'bad4' may not have specific procedures 's4a' and 's4b' as their interfaces are not distinguishable
  interface BAD4
    subroutine S4A(W, X, Y, Z)
      real :: W, Y
      integer :: X, Z
    end subroutine
    subroutine S4B(X, W, Z, Y)
      real :: X, Y
      integer :: W, Z
    end subroutine
  end interface
end
module m4b
  !ERROR: Generic 'bad4' may not have specific procedures 's4b' and 's4a' as their interfaces are not distinguishable
  interface BAD4
    subroutine S4B(X, W, Z, Y)
      real :: X, Y
      integer :: W, Z
    end subroutine
    subroutine S4A(W, X, Y, Z)
      real :: W, Y
      integer :: X, Z
    end subroutine
  end interface
end

! C.10.6(19)
module m5
  interface GOOD5
    subroutine S5A(X)
      real :: X
    end subroutine
    subroutine S5B(Y, X)
      real :: Y, X
    end subroutine
  end interface
end

module FRUITS
  type :: FRUIT
  end type
  type, extends(FRUIT) :: APPLE
  end type
  type, extends(FRUIT) :: PEAR
  end type
  type, extends(PEAR) :: BOSC
  end type
end

! C.10.6(21)
! type(PEAR) :: A_PEAR
! type(BOSC) :: A_BOSC
! BAD6(A_PEAR,A_BOSC)  ! could be s6a or s6b
module m6
  !ERROR: Generic 'bad6' may not have specific procedures 's6a' and 's6b' as their interfaces are not distinguishable
  interface BAD6
    subroutine S6A(X, Y)
      use FRUITS
      class(PEAR) :: X, Y
    end subroutine
    subroutine S6B(X, Y)
      use FRUITS
      class(FRUIT) :: X
      class(BOSC) :: Y
    end subroutine
  end interface
end
module m6b
  !ERROR: Generic 'bad6' may not have specific procedures 's6b' and 's6a' as their interfaces are not distinguishable
  interface BAD6
    subroutine S6B(X, Y)
      use FRUITS
      class(FRUIT) :: X
      class(BOSC) :: Y
    end subroutine
    subroutine S6A(X, Y)
      use FRUITS
      class(PEAR) :: X, Y
    end subroutine
  end interface
end

! C.10.6(22)
module m7
  interface GOOD7
    subroutine S7A(X, Y, Z)
      use FRUITS
      class(PEAR) :: X, Y, Z
    end subroutine
    subroutine S7B(X, Z, W)
      use FRUITS
      class(FRUIT) :: X
      class(BOSC) :: Z
      class(APPLE), optional :: W
    end subroutine
  end interface
end
module m7b
  interface GOOD7
    subroutine S7B(X, Z, W)
      use FRUITS
      class(FRUIT) :: X
      class(BOSC) :: Z
      class(APPLE), optional :: W
    end subroutine
    subroutine S7A(X, Y, Z)
      use FRUITS
      class(PEAR) :: X, Y, Z
    end subroutine
  end interface
end

! C.10.6(25)
! Invalid generic (according to the rules), despite the fact that it is unambiguous
module m8
  !ERROR: Generic 'bad8' may not have specific procedures 's8a' and 's8b' as their interfaces are not distinguishable
  interface BAD8
    subroutine S8A(X, Y, Z)
      real, optional :: X
      integer :: Y
      real :: Z
    end subroutine
    subroutine S8B(X, Z, Y)
      integer, optional :: X
      integer :: Z
      real :: Y
    end subroutine
  end interface
end
