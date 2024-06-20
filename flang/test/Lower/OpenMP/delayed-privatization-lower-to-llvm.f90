! Tests the OMPIRBuilder can handle multiple privatization regions that contain
! multiple BBs (for example, for allocatables).

! RUN: %flang -S -emit-llvm -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %s 2>&1 | FileCheck %s

! CHECK: @[[SOURCE:.*]] = linkonce constant [{{.*}} x i8] c"{{.*}}delayed-privatization-lower-to-llvm.f90\00", comdat

subroutine lower_allocatable(x)
  integer, allocatable :: x, y
!$omp parallel private(x, y)
  x = y
!$omp end parallel
end

! CHECK-LABEL: define void @lower_allocatable_
! CHECK:         ret void
! CHECK-NEXT:  }

! CHECK-LABEL: define internal void @lower_allocatable_..omp_par
! CHECK-DAG:     call ptr @malloc
! CHECK-DAG:     call ptr @malloc
! CHECK-DAG:     call void @free
! CHECK-DAG:     call void @free
! CHECK:       }

subroutine lower_region_with_if_print
  real(kind=8), dimension(1,1) :: u1
  !$omp parallel firstprivate(u1) 
    if (any(u1/=1)) u1 = u1 + 1
  !$omp end parallel
end subroutine

! CHECK-LABEL: define void @lower_region_with_if_print_
! CHECK:         ret void
! CHECK-NEXT:  }

! CHECK-LABEL: define internal void @lower_region_with_if_print_..omp_par
! CHECK:         call i1 @_FortranAAny(ptr %{{[^[:space:]]+}}, ptr @[[SOURCE]], i32 {{.*}}, i32 {{.*}})
! CHECK:       }
