! RUN: %python %S/test_errors.py %s %flang_fc1
pure subroutine s1
 contains
  !ERROR: An internal subprogram of a pure subprogram must also be pure
  subroutine t1
  end
  pure subroutine t2 ! ok
  end
  elemental subroutine t3(k) ! ok
    integer, intent(in) :: k
  end
  !ERROR: An internal subprogram of a pure subprogram must also be pure
  impure elemental subroutine t4(k)
    integer, intent(in) :: k
  end
  !ERROR: An internal subprogram of a pure subprogram must also be pure
  elemental impure subroutine t5(k)
    integer, intent(in) :: k
  end
end

elemental subroutine s2(j)
  integer, intent(in) :: j
 contains
  !ERROR: An internal subprogram of a pure subprogram must also be pure
  subroutine t1
  end
  pure subroutine t2 ! ok
  end
  elemental subroutine t3(k) ! ok
    integer, intent(in) :: k
  end
  !ERROR: An internal subprogram of a pure subprogram must also be pure
  impure elemental subroutine t4(k)
    integer, intent(in) :: k
  end
  !ERROR: An internal subprogram of a pure subprogram must also be pure
  elemental impure subroutine t5(k)
    integer, intent(in) :: k
  end
end

impure elemental subroutine s3(j)
  integer, intent(in) :: j
 contains
  subroutine t1
  end
  pure subroutine t2
  end
  elemental subroutine t3(k)
    integer, intent(in) :: k
  end
  impure elemental subroutine t4(k)
    integer, intent(in) :: k
  end
  elemental impure subroutine t5(k)
    integer, intent(in) :: k
  end
end
