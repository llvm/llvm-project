! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
module m1
  type a1
     integer ::x1=1
  end type a1
  type,extends(a1)::a2
     integer ::x2=3
  end type a2
  type,extends(a2)::a3
     integer   ::x3=3
  end type a3
end module m1

program test
  use m1
  type(a3) v
  !PORTABILITY: Whole parent component 'a2' in structure constructor should not be anonymous
  v=a3(a2(x1=18,x2=6),x3=6)
  !PORTABILITY: Whole parent component 'a1' in structure constructor should not be anonymous
  v=a3(a1(x1=18),x2=6,x3=6)
  !PORTABILITY: Whole parent component 'a2' in structure constructor should not be anonymous
  !PORTABILITY: Whole parent component 'a1' in structure constructor should not be anonymous
  v=a3(a2(a1(x1=18),x2=6),x3=6)
  v=a3(a2=a2(a1=a1(x1=18),x2=6),x3=6) ! ok
end
