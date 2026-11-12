!RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
module m
  interface generic1 ! ok
    module procedure :: sub1
  end interface
  !ERROR: Generic 'generic2' may not have specific procedures 'sub1' and 'sub2' as their interfaces are not distinguishable
  interface generic2
    module procedure :: sub1, sub2
  end interface
 contains
  subroutine sub1(a,len)
    type(*), intent(in) :: a(*)
    integer len
    print *, 'in sub'
  end
  subroutine sub2(a,len)
    character(*), intent(in) :: a
    integer len
    print *, 'in sub2'
  end
end

program test
  use m
  character(3) :: foo = "abc"
  !PORTABILITY: A scalar actual argument for an assumed-size TYPE(*) dummy is not portable [-Wassumed-type-size-dummy]
  call sub1(foo, 3) ! ok
  !PORTABILITY: A scalar actual argument for an assumed-size TYPE(*) dummy is not portable [-Wassumed-type-size-dummy]
  call generic1(foo, 3) ! ok
  !ERROR: The actual arguments to the generic procedure 'generic2' matched multiple specific procedures, perhaps due to use of NULL() without MOLD= or an actual procedure with an implicit interface
  call generic2(foo, 3)
end
