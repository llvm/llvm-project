! RUN: %flang_fc1 -fopenmp -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s

        program name_with_space
!CHECK: !$OMP THREADPRIVATE(/cc/, var1)
!$omp threadprivate(/c c/, var 1)

!CHECK: !$OMP PARALLEL PRIVATE(somevar,expr1,expr2) IF(expr2>expr1)
!$omp parallel private(some var, expr 1, ex pr2)
!$omp+ if (exp r2 > ex pr1)
!$omp critical (x_x)
        print '(a)', 'Hello World'
!CHECK: !$OMP END CRITICAL(x_x)
!$omp end critical (x _x)
!$omp end parallel
        end program name_with_space
