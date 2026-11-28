! RUN: %python %S/test_modfile.py %s %flang_fc1
! References to __builtin_selected_real_kind, and named constants depending on
! them, are not folded in the intrinsic modules until the module file is read.
module iso_fortran_env_impl
  intrinsic :: __builtin_selected_real_kind
  integer, parameter :: k = __builtin_selected_real_kind(6, 37)
  integer, parameter :: safe = merge(k, -1, k >= 0)
  integer, parameter :: kinds(*) = [pack([k], k >= 0)]
  integer, parameter :: folded = selected_real_kind(6, 37)
end

!Expect: iso_fortran_env_impl.mod
!module iso_fortran_env_impl
!intrinsic::__builtin_selected_real_kind
!integer(4),parameter::k=__builtin_selected_real_kind(6_4,37_4)
!integer(4),parameter::safe=merge(k,-1_4,k>=0_4)
!intrinsic::merge
!integer(4),parameter::kinds(1_8:*)=[INTEGER(4)::pack([INTEGER(4)::k],k>=0_4)]
!intrinsic::pack
!integer(4),parameter::folded=4_4
!intrinsic::selected_real_kind
!end
