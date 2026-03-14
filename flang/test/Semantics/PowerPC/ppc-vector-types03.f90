! RUN: %python %S/../test_errors.py %s %flang_fc1
! REQUIRES: target=aarch64{{.*}} || target=arm{{.*}} || target=x86{{.*}}

program ppc_vec_types04
  implicit none
!ERROR: Vector type is only supported for PowerPC
!ERROR: No explicit type declared for 'vi'
  vector(integer(4)) :: vi
!ERROR: Vector type is only supported for PowerPC
!ERROR: No explicit type declared for 'vr'
  vector(real(8)) :: vr
!ERROR: Vector type is only supported for PowerPC
!ERROR: No explicit type declared for 'vu'
  vector(unsigned(2)) :: vu
!ERROR: Vector type is only supported for PowerPC
!ERROR: No explicit type declared for 'vp'
  __vector_pair :: vp
!ERROR: Vector type is only supported for PowerPC
!ERROR: No explicit type declared for 'vq'
  __vector_quad :: vq
end program ppc_vec_types04
