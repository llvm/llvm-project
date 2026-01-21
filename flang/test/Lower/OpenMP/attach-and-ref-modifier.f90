! RUN: %flang_fc1 -fopenmp -fopenmp-version=61 -emit-hlfir %s -o - 2>&1 | FileCheck %s

subroutine attach_always()
    integer, pointer :: x

!CHECK: {{.*}} = omp.map.info{{.*}}map_clauses(tofrom, attach_always){{.*}}
    !$omp target map(attach(always): x)
        x = 1
    !$omp end target
end

subroutine attach_never()
    integer, pointer :: x

!CHECK: {{.*}} = omp.map.info{{.*}}map_clauses(tofrom, attach_never){{.*}}
    !$omp target map(attach(never): x)
        x = 1
    !$omp end target
end

subroutine attach_auto()
    integer, pointer :: x
!CHECK: {{.*}} = omp.map.info{{.*}}map_clauses(tofrom, attach_auto){{.*}}
    !$omp target map(attach(auto): x)
        x = 1
    !$omp end target
end

subroutine ref_ptr_ptee()
    integer, pointer :: x

!CHECK: {{.*}} = omp.map.info{{.*}}map_clauses(to, ref_ptr_ptee){{.*}}
    !$omp target map(ref_ptr_ptee, to: x)
        x = 1
    !$omp end target
end

subroutine ref_ptr()
  integer, pointer :: x

!CHECK: {{.*}} = omp.map.info{{.*}}map_clauses(to, ref_ptr){{.*}}
    !$omp target map(ref_ptr, to: x)
        x = 1
    !$omp end target
end

subroutine ref_ptee()
  integer, pointer :: x

!CHECK: {{.*}} = omp.map.info{{.*}}map_clauses(to, ref_ptee){{.*}}
    !$omp target map(ref_ptee, to: x)
        x = 1
    !$omp end target
end

subroutine ref_ptr_ptee_attach_never()
    integer, pointer :: x

!CHECK: {{.*}} = omp.map.info{{.*}}map_clauses(to, attach_never, ref_ptr_ptee){{.*}}
    !$omp target map(attach(never), ref_ptr_ptee, to: x)
        x = 1
    !$omp end target
end
