! RUN: %python %S/test_errors.py %s %flang_fc1

module m
    type :: t1
      real :: x
    end type
    type :: t2(k)
      integer, kind :: k
      real(kind=k) :: x
    end type
    type :: t3
      real :: x
    end type
    type, extends(t1) :: t4
      integer :: y
    end type
    type :: t5
      sequence
      integer :: x
      integer :: y
    end type
  
  
    integer :: i
    real :: r
    type(t1) :: x1, y1
    type(t2(4)) :: x24, y24
    type(t2(8)) :: x28
    type(t3) :: x3
    type(t4) :: x4
    type(t5) :: x5
    class(t1), allocatable :: a1
    class(t3), allocatable :: a3

  
    logical :: t1_1 = same_type_as(x1, x1)
    logical :: t1_2 = same_type_as(x1, y1)
    logical :: t1_3 = same_type_as(x24, x24)
    logical :: t1_4 = same_type_as(x24, y24)
    logical :: t1_5 = same_type_as(x24, x28) ! ignores parameter
    logical :: t1_6 = .not. same_type_as(x1, x3)
    logical :: t1_7 = .not. same_type_as(a1, a3)
    !ERROR: Actual argument for 'a=' has bad type 't5', expected extensible derived or unlimited polymorphic type
    logical :: t1_8 = same_type_as(x5, x5)
    !ERROR: Actual argument for 'a=' has bad type 't5', expected extensible derived or unlimited polymorphic type
    logical :: t1_9 = same_type_as(x5, x1)
    !ERROR: Actual argument for 'b=' has bad type 't5', expected extensible derived or unlimited polymorphic type
    logical :: t1_10 = same_type_as(x1, x5)
    !ERROR: Actual argument for 'a=' has bad type 'INTEGER(4)', expected extensible derived or unlimited polymorphic type
    logical :: t1_11 = same_type_as(i, i)
    !ERROR: Actual argument for 'a=' has bad type 'REAL(4)', expected extensible derived or unlimited polymorphic type
    logical :: t1_12 = same_type_as(r, r)
    !ERROR: Actual argument for 'a=' has bad type 'INTEGER(4)', expected extensible derived or unlimited polymorphic type
    logical :: t1_13 = same_type_as(i, t)
    
    logical :: t2_1 = extends_type_of(x1, y1)
    logical :: t2_2 = extends_type_of(x24, x24)
    logical :: t2_3 = extends_type_of(x24, y24)
    logical :: t2_4 = extends_type_of(x24, x28) ! ignores parameter
    logical :: t2_5 = .not. extends_type_of(x1, x3)
    logical :: t2_6 = .not. extends_type_of(a1, a3)
    logical :: t2_7 = .not. extends_type_of(x1, x4)
    logical :: t2_8 = extends_type_of(x4, x1)
    !ERROR: Actual argument for 'a=' has bad type 't5', expected extensible derived or unlimited polymorphic type
    logical :: t2_9 = extends_type_of(x5, x5)
    !ERROR: Actual argument for 'a=' has bad type 't5', expected extensible derived or unlimited polymorphic type
    logical :: t2_10 = extends_type_of(x5, x1)
    !ERROR: Actual argument for 'mold=' has bad type 't5', expected extensible derived or unlimited polymorphic type
    logical :: t2_11 = extends_type_of(x1, x5)
end module
  