! RUN: %python %S/test_modfile.py %s %flang_fc1
module m1
  type foo
  end type
  interface foo
  end interface
end

!Expect: m1.mod
!module m1
!type::foo
!end type
!interface foo
!end interface
!end

module m2
  use m1, only: bar => foo
end

!Expect: m2.mod
!module m2
!use m1,only:bar=>foo
!use m1,only:bar=>foo
!interface bar
!end interface
!end

module m3
 contains
  subroutine sub(x)
    use m2
    type(bar) x
  end
end

!Expect: m3.mod
!module m3
!contains
!subroutine sub(x)
!use m2,only:bar
!type(bar)::x
!end
!end
