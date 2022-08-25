! RUN: %python %S/test_errors.py %s %flang_fc1
1 program main
1  type one
2    real x
3  end type one
1  type two
2    real x
     !ERROR: Label '2' is not distinct
2    real y
3  end type two
2  interface
2   subroutine sub1(p, q)
3    interface
3     subroutine p
4     end subroutine
3     subroutine q
4     end subroutine
4    end interface
5   end subroutine
2   subroutine sub2(p, q)
3    interface
3     subroutine p
4     end subroutine
3     subroutine q
4     end subroutine
4    end interface
5   end subroutine
3  end interface
4  call sub3
5 contains
1  subroutine sub3
2   continue
3   block
     !ERROR: Label '2' is not distinct
2    continue
4   end block
5  end subroutine
6 end program
