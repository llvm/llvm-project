! RUN: %python %S/test_modfile.py %s %flang_fc1
module m1
  type foo
    integer :: c1 = 123
  end type
end

module m2
  use m1, only: foo
  type baz
    type(foo) :: d = foo()
  end type
  type bar
    type(baz) :: e = baz()
  end type
end

module m3
  use m1, only: m1foo => foo
  type foo
    type(m1foo), private :: c2 = m1foo()
  end type
end

module m4
  use m2, only: m3foo => foo
  type foo
    type(m3foo), private :: c3 = m3foo()
  end type
end

module m5
  use m2, only: m2bar => bar
  use m4, only: foo
  type blah
    type(m2bar) :: f = m2bar()
  end type
end

!Expect: m1.mod
!module m1
!type::foo
!integer(4)::c1=123_4
!end type
!end

!Expect: m2.mod
!module m2
!use m1,only:foo
!type::baz
!type(foo)::d=foo(c1=123_4)
!end type
!type::bar
!type(baz)::e=baz(d=foo(c1=123_4))
!end type
!end

!Expect: m3.mod
!module m3
!use m1,only:m1foo=>foo
!type::foo
!type(m1foo),private::c2=m1foo(c1=123_4)
!end type
!end

!Expect: m4.mod
!module m4
!use m2,only:m3foo=>foo
!type::foo
!type(m3foo),private::c3=m3foo(c1=123_4)
!end type
!end

!Expect: m5.mod
!module m5
!use m2,only:m2$foo=>foo
!use m2,only:baz
!use m2,only:m2bar=>bar
!use m4,only:foo
!private::m2$foo
!private::baz
!type::blah
!type(m2bar)::f=m2bar(e=baz(d=m2$foo(c1=123_4)))
!end type
!end
