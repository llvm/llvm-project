! RUN: %python %S/../test_errors.py %s %flang -fopenmp
!
! OpenMP 5.2: 5.2 threadprivate directive restriction

subroutine task_untied01()
    integer, save :: var_01, var_02(2)
    real          :: var_03
    common /c/ var_03

    !$omp threadprivate(var_01, var_02)
    !$omp threadprivate(/c/)

    !$omp task untied
        !ERROR: A THREADPRIVATE variable `var_01` cannot appear in an UNTIED TASK region
        var_01    = 10
        !ERROR: A THREADPRIVATE variable `var_02` cannot appear in an UNTIED TASK region
        !ERROR: A THREADPRIVATE variable `var_01` cannot appear in an UNTIED TASK region
        var_02(1) = sum([var_01, 20])
    !$omp end task

    !$omp task untied
        !ERROR: A THREADPRIVATE variable `var_02` cannot appear in an UNTIED TASK region
        !ERROR: A THREADPRIVATE variable `var_02` cannot appear in an UNTIED TASK region
        var_02(2) = product(var_02)
        !ERROR: A THREADPRIVATE variable `var_03` cannot appear in an UNTIED TASK region
        var_03    = 3.14
    !$omp end task
end subroutine task_untied01
