! RUN: %python %S/test_modfile.py %s %flang_fc1
! Simplified regression test for crashreported in
! https://github.com/llvm/llvm-project/issues/123534.
module m
  interface
    ! f1 returns a pointer to a procedure whose result characteristics
    ! depend on the value of a dummy argument.
    function f1()
      interface
        function f2(n)
          integer, intent(in) :: n
          character(n), pointer :: f2
        end
      end interface
      procedure (f2), pointer :: f1
    end
  end interface
end

!Expect: m.mod
!module m
!interface
!function f1()
!interface
!function f2(n)
!integer(4),intent(in)::n
!character(n,1),pointer::f2
!end
!end interface
!procedure(f2),pointer::f1
!end
!end interface
!end
