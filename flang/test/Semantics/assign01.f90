! RUN: %python %S/test_errors.py %s %flang_fc1
! 10.2.3.1(2) All masks and LHS of assignments in a WHERE must conform

subroutine s1
  integer :: a1(10), a2(10)
  logical :: m1(10), m2(5,5)
  m1 = .true.
  m2 = .false.
  a1 = [((i),i=1,10)]
  where (m1)
    a2 = 1
  !ERROR: Must have rank 1 to match prior mask or assignment of WHERE construct
  elsewhere (m2)
    a2 = 2
  elsewhere
    a2 = 3
  end where
end

subroutine s2
  logical, allocatable :: m1(:), m4(:,:)
  logical :: m2(2), m3(3)
  where(m1)
    where(m2)
    end where
    !ERROR: Dimension 1 must have extent 2 to match prior mask or assignment of WHERE construct
    where(m3)
    end where
    !ERROR: Must have rank 1 to match prior mask or assignment of WHERE construct
    where(m4)
    end where
  endwhere
  where(m1)
    where(m3)
    end where
  !ERROR: Dimension 1 must have extent 3 to match prior mask or assignment of WHERE construct
  elsewhere(m2)
  end where
end

subroutine s3
  logical, allocatable :: m1(:,:)
  logical :: m2(4,2)
  real :: x(4,4), y(4,4)
  real :: a(4,5), b(4,5)
  where(m1)
    x = y
    !ERROR: Dimension 2 must have extent 4 to match prior mask or assignment of WHERE construct
    a = b
    !ERROR: Dimension 2 must have extent 4 to match prior mask or assignment of WHERE construct
    where(m2)
    end where
  end where
end

subroutine s4
  integer :: x1 = 0, x2(2) = 0
  logical :: l1 = .false., l2(2) = (/.true., .false./), l3 = .false.
  !ERROR: The mask or variable must not be scalar
  where (l1)
    !ERROR: The mask or variable must not be scalar
    x1 = 1
  end where
  !ERROR: The mask or variable must not be scalar
  where (l1)
    !ERROR: The mask or variable must not be scalar
    where (l3)
      !ERROR: The mask or variable must not be scalar
      x1 = 1
    end where
  end where
  !ERROR: The mask or variable must not be scalar
  where (l2(2))
    !ERROR: The mask or variable must not be scalar
    x2(2) = 1
  end where
end
