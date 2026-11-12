! RUN: %python %S/test_symbols.py %s %flang_fc1
! Regression test of name resolution bug
!DEF: /m Module
module m
 !DEF: /m/base ABSTRACT, PUBLIC DerivedType
 type, abstract :: base
 contains
  !DEF: /m/base/foo Generic
  !DEF: /m/base/spec DEFERRED ProcBinding
  generic :: foo => spec
  !DEF: /m/iface ABSTRACT, PUBLIC (Subroutine) Subprogram
  !REF: /m/base/spec
  procedure(iface), deferred :: spec
 end type
 abstract interface
  !REF: /m/iface
  !DEF: /m/iface/this ObjectEntity CLASS(base)
  subroutine iface (this)
   !REF: /m/base
   import :: base
   !REF: /m/base
   !REF: /m/iface/this
   class(base) :: this
  end subroutine
 end interface
 !REF: /m/base
 !DEF: /m/ext PUBLIC DerivedType
 type, extends(base) :: ext
 contains
  !DEF: /m/ext/spec ProcBinding
  !DEF: /m/foo PUBLIC (Subroutine) Subprogram
  procedure :: spec => foo
 end type
contains
 !REF: /m/foo
 !DEF: /m/foo/this ObjectEntity CLASS(ext)
 subroutine foo (this)
  !REF: /m/ext
  !REF: /m/foo/this
  class(ext) :: this
 end subroutine
end module
