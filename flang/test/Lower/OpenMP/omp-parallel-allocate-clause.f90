! This test checks lowering of OpenMP parallel Directive with
! `PRIVATE` clause present.

! RUN: %bbc -fopenmp -emit-fir %s -o  - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

!FIRDialect: func @_QPallocate_clause(%arg0: !fir.ref<f32>, %arg1: !fir.ref<f32>) {
!FIRDialect-DAG: %[[A:.*]] = fir.alloca i32 {name = "{{.*}}Ea"}
!FIRDialect-DAG: %[[B:.*]] = fir.alloca i32 {name = "{{.*}}Eb"}
!FIRDialect-DAG: %[[C:.*]] = fir.alloca i32 {name = "{{.*}}Ec"}
!FIRDialect-DAG: %[[D:.*]] = fir.alloca i32 {name = "{{.*}}Ed"}
!FIRDialect-DAG: %[[E:.*]] = fir.alloca i32 {name = "{{.*}}Ee"}
!FIRDialect-DAG: %[[F:.*]] = fir.alloca i32 {name = "{{.*}}Ef"}
!FIRDialect-DAG: %[[G:.*]] = fir.alloca i32 {name = "{{.*}}Eg"}
!FIRDialect-DAG: %[[H:.*]] = fir.alloca i32 {name = "{{.*}}Eh"}
!FIRDialect-DAG: %[[X:.*]] = fir.alloca i32 {name = "{{.*}}Ex"}
!FIRDialect-DAG: %[[Y:.*]] = fir.alloca i32 {name = "{{.*}}Ey"}
!FIRDialect-DAG: %[[Z:.*]] = fir.alloca i32 {name = "{{.*}}Ez"}

subroutine allocate_clause(arg1, arg2)
use omp_lib
        integer :: x, y, z, a, b, c, d, e, f, g, h

!FIRDialect-DAG: %[[LARGE:.*]] = constant 2 : i32
!FIRDialect-DAG: %[[DEFAULT:.*]] = constant 1 : i32
!FIRDialect-DAG: %[[CONSTANT:.*]] = constant 3 : i32

!FIRDialect: omp.parallel private(%[[X]] : !fir.ref<i32>, %[[Y]] :
!fir.ref<i32>, %[[Z]] : !fir.ref<i32>, %[[A]] : !fir.ref<i32>, %[[B]] :
!fir.ref<i32>, %[[C]] : !fir.ref<i32>, %[[D]] :
!fir.ref<i32>) allocate(%[[LARGE]] : i32 -> %[[X]] :
!fir.ref<i32>, %[[LARGE]] : i32 -> %[[Y]] :
!fir.ref<i32>, %[[LARGE]] : i32 -> %[[Z]] :
!fir.ref<i32>, %[[DEFAULT]] : i32 -> %[[A]] :
!fir.ref<i32>, %[[DEFAULT]] : i32 -> %[[B]] :
!fir.ref<i32>, %[[CONSTANT]] : i32 -> %[[C]] :
!fir.ref<i32>, %[[CONSTANT]] : i32 -> %[[D]] : !fir.ref<i32>) {
!FIRDialect:    omp.terminator
!FIRDialect:  }

!$OMP PARALLEL PRIVATE(x, y, z, a, b, c, d) ALLOCATE(omp_large_cap_mem_alloc: x, y, z) ALLOCATE(a, b) ALLOCATE(omp_const_mem_alloc: c, d)
        print*, "ALLOCATE"
        print*, x, y, z
!$OMP END PARALLEL

!FIRDialect-DAG: %[[DEFAULT2:.*]] = constant 1 : i32
!FIRDialect: omp.parallel private(%[[E]] : !fir.ref<i32>, %[[F]] :
!fir.ref<i32>) allocate(%[[DEFAULT2]] : i32 -> %[[E]] :
!fir.ref<i32>, %[[DEFAULT2]] : i32 -> %[[F]] : !fir.ref<i32>) {
!FIRDialect:    omp.terminator
!FIRDialect:  }

!$OMP PARALLEL PRIVATE(e, f) ALLOCATE(e, f)
        print*, "ALLOCATE"
        print*, x, y, z
!$OMP END PARALLEL

!FIRDialect-DAG: %[[LOWLAT:.*]] = constant 5 : i32
!FIRDialect: omp.parallel private(%[[G]] : !fir.ref<i32>, %[[H]] :
!fir.ref<i32>) allocate(%[[LOWLAT]] : i32 -> %[[G]] :
!fir.ref<i32>, %[[LOWLAT]] : i32 -> %[[H]] : !fir.ref<i32>) {
!FIRDialect:    omp.terminator
!FIRDialect:  }

!$OMP PARALLEL PRIVATE(g, h) ALLOCATE(omp_low_lat_mem_alloc : g, h)
        print*, "ALLOCATE"
        print*, x, y, z
!$OMP END PARALLEL

end subroutine
