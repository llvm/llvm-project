! RUN: %python %S/test_symbols.py %s %flang_fc1
!DEF: /m Module
module m
 !DEF: /m/t PUBLIC DerivedType
 type :: t
 contains
  !DEF: /m/forwardreferenced ELEMENTAL, IMPURE, MODULE, PUBLIC (Subroutine) Subprogram
  final :: forwardreferenced
 end type
 interface
  !REF: /m/forwardreferenced
  !DEF: /m/forwardreferenced/this INTENT(INOUT) ObjectEntity TYPE(t)
  impure elemental module subroutine forwardreferenced (this)
   !REF: /m/t
   !REF: /m/forwardreferenced/this
   type(t), intent(inout) :: this
  end subroutine
 end interface
end module
