!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %flang_fc1 -fopenmp -emit-llvm %s -o - | FileCheck %s

! Regression test for https://github.com/llvm/llvm-project/issues/112545
! Test if OMPIRBuilder passes the correct ident_f->flags for worksharing constructs. Intended:
! DO: ident_t->flags == 0x200 (KMP_IDENT_WORK_LOOP) | 0x002 (KMP_IDENT_KMPC)
! SECTIONS/SECTION: ident_t->flags == 0x200 (KMP_IDENT_WORK_LOOP) | 0x002 (KMP_IDENT_KMPC)
! DISTRIBUTE: ident_t->flags == 0x800 (KMP_IDENT_WORK_DISTRIBUTE) | 0x002 (KMP_IDENT_KMPC)
! DISTRIBUTE DO:
!  ident_t->flags == 0x800 (KMP_IDENT_WORK_DISTRIBUTE) | 0x200 (KMPC_IDENT_WORK_LOOP | 0x002 (KMP_IDENT_KMPC)

subroutine workshare_do_ident_flag()
  integer :: i

  !$OMP PARALLEL
  !$OMP DO
  do i = 1, 10
  end do
  !$OMP END DO
  !$OMP END PARALLEL
end subroutine workshare_do_ident_flag

subroutine workshare_sections_ident_flag()
  !$OMP PARALLEL
  !$OMP SECTIONS
  !$OMP SECTION
  block
  end block
  !$OMP END SECTIONS
  !$OMP END PARALLEL
end subroutine workshare_sections_ident_flag

subroutine workshare_distribute_ident_flag()
  integer :: i

  !$OMP TEAMS
  !$OMP DISTRIBUTE
  do i = 1, 10
  end do
  !$OMP END DISTRIBUTE
  !$OMP END TEAMS
end subroutine workshare_distribute_ident_flag

subroutine workshare_distribute_do_ident_flag()
  integer :: i

  !$OMP TEAMS
  !$OMP DISTRIBUTE PARALLEL DO
  do i = 1, 10
  end do
  !$OMP END DISTRIBUTE PARALLEL DO
  !$OMP END TEAMS
end subroutine workshare_distribute_do_ident_flag

! CHECK: @[[IDENT_DO:[0-9]+]] = private unnamed_addr constant %struct.ident_t { i32 {{[0-9]+}}, i32 514, i32 {{[0-9]+}}, i32 {{[0-9]+}}, ptr @0 }, align 8
! CHECK: @[[IDENT_DISTRIBUTE:[0-9]+]] = private unnamed_addr constant %struct.ident_t { i32 {{[0-9]+}}, i32 2050, i32 {{[0-9]+}}, i32 {{[0-9]+}}, ptr @0 }, align 8
! CHECK: @[[IDENT_DISTRIBUTE_DO:[0-9]+]] = private unnamed_addr constant %struct.ident_t { i32 {{[0-9]+}}, i32 2562, i32 {{[0-9]+}}, i32 {{[0-9]+}}, ptr @0 }, align 8

! Test workshare_do_ident_flag
! CHECK: call void @__kmpc_for_static_init_{{.*}}(ptr @[[IDENT_DO]], {{.*}})
! Test workshare_sections_ident_flag
! CHECK: call void @__kmpc_for_static_init_{{.*}}(ptr @[[IDENT_DO]], {{.*}})
! Test workshare_distribute_ident_flag
! CHECK: call void @__kmpc_for_static_init_{{.*}}(ptr @[[IDENT_DISTRIBUTE]], {{.*}})
! Test workshare_distribute_do_ident_flag
! CHECK: call void @__kmpc_dist_for_static_init_{{.*}}(ptr @[[IDENT_DISTRIBUTE_DO]], {{.*}})
