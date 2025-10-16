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

    integer(kind=merge(kind(1),-1,same_type_as(x1, x1))) same_type_as_x1_x1_true
    integer(kind=merge(kind(1),-1,same_type_as(x1, y1))) same_type_as_x1_y1_true
    integer(kind=merge(kind(1),-1,same_type_as(x24, x24))) same_type_as_x24_x24_true
    integer(kind=merge(kind(1),-1,same_type_as(x24, y24))) same_type_as_x24_y24_true
    integer(kind=merge(kind(1),-1,same_type_as(x24, x28))) same_type_as_x24_x28_true
    !ERROR: INTEGER(KIND=-1) is not a supported type
    integer(kind=merge(kind(1),-1,same_type_as(x1, x3))) same_type_as_x1_x3_false
    !ERROR: INTEGER(KIND=-1) is not a supported type
    integer(kind=merge(kind(1),-1,same_type_as(a1, a3))) same_type_as_a1_a3_false
    !ERROR: Actual argument for 'a=' has type 't5', but was expected to be an extensible or unlimited polymorphic type
    logical :: t1_8 = same_type_as(x5, x5)
    !ERROR: Actual argument for 'a=' has type 't5', but was expected to be an extensible or unlimited polymorphic type
    logical :: t1_9 = same_type_as(x5, x1)
    !ERROR: Actual argument for 'b=' has type 't5', but was expected to be an extensible or unlimited polymorphic type
    logical :: t1_10 = same_type_as(x1, x5)
    !ERROR: Actual argument for 'a=' has bad type 'INTEGER(4)', expected extensible or unlimited polymorphic type
    logical :: t1_11 = same_type_as(i, i)
    !ERROR: Actual argument for 'a=' has bad type 'REAL(4)', expected extensible or unlimited polymorphic type
    logical :: t1_12 = same_type_as(r, r)
    !ERROR: Actual argument for 'a=' has bad type 'INTEGER(4)', expected extensible or unlimited polymorphic type
    logical :: t1_13 = same_type_as(i, t)

    integer(kind=merge(kind(1),-1,extends_type_of(x1, y1))) extends_type_of_x1_y1_true
    integer(kind=merge(kind(1),-1,extends_type_of(x24, x24))) extends_type_of_x24_x24_true
    integer(kind=merge(kind(1),-1,extends_type_of(x24, y24))) extends_type_of_x24_y24_true
    integer(kind=merge(kind(1),-1,extends_type_of(x24, x28))) extends_type_of_x24_x28_true
    !ERROR: INTEGER(KIND=-1) is not a supported type
    integer(kind=merge(kind(1),-1,extends_type_of(x1, x3))) extends_type_of_x1_x3_false
    !ERROR: INTEGER(KIND=-1) is not a supported type
    integer(kind=merge(kind(1),-1,extends_type_of(a1, a3))) extends_type_of_a1_a3_false
    !ERROR: INTEGER(KIND=-1) is not a supported type
    integer(kind=merge(kind(1),-1,extends_type_of(x1, x4))) extends_type_of_x1_x4_false
    integer(kind=merge(kind(1),-1,extends_type_of(x4, x1))) extends_type_of_x4_x1_true
    !ERROR: Actual argument for 'a=' has type 't5', but was expected to be an extensible or unlimited polymorphic type
    logical :: t2_9 = extends_type_of(x5, x5)
    !ERROR: Actual argument for 'a=' has type 't5', but was expected to be an extensible or unlimited polymorphic type
    logical :: t2_10 = extends_type_of(x5, x1)
    !ERROR: Actual argument for 'mold=' has type 't5', but was expected to be an extensible or unlimited polymorphic type
    logical :: t2_11 = extends_type_of(x1, x5)
end module
