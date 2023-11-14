! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Confirm enforcement of constraint C723 in F2018 for procedure pointers

      module m
       contains
        subroutine subr(parg)
          !PORTABILITY: A dummy procedure pointer should not have assumed-length CHARACTER(*) result type
          procedure(character(*)), pointer :: parg
          !ERROR: An assumed (*) type parameter may be used only for a (non-statement function) dummy argument, associate name, character named constant, or external function result
          procedure(character(*)), pointer :: plocal
          print *, parg()
          plocal => parg
          call subr_1(plocal)
        end subroutine

        subroutine subr_1(parg_1)
          !PORTABILITY: A dummy procedure pointer should not have assumed-length CHARACTER(*) result type
          procedure(character(*)), pointer :: parg_1
          print *, parg_1()
        end subroutine
      end module

      character(*) function f()
        f = 'abcdefgh'
      end function

      program test
        use m
        character(4), external :: f
        procedure(character(4)), pointer :: p
        p => f
        call subr(p)
      end

