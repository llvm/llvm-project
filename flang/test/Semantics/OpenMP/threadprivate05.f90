! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.1
! Check OpenMP construct validity for the following directives:
! 2.21.2 Threadprivate Directive

module mod0
  integer :: mi

contains
  subroutine subm()
    integer, save :: mmi

    !ERROR: The THREADPRIVATE directive and the common block or variable in it must appear in the same declaration section of a scoping unit
    !$omp threadprivate(mi)
    mi = 1
  contains
    subroutine subsubm()
      !ERROR: The THREADPRIVATE directive and the common block or variable in it must appear in the same declaration section of a scoping unit
      !$omp threadprivate(mmi)
    end
  end
end

module mod1
  integer :: mod_i
end

program main
  use mod1
  integer, save :: i
  integer :: j

  !ERROR: The THREADPRIVATE directive and the common block or variable in it must appear in the same declaration section of a scoping unit
  !$omp threadprivate(mod_i)

contains
  subroutine sub()
    !ERROR: The THREADPRIVATE directive and the common block or variable in it must appear in the same declaration section of a scoping unit
    !ERROR: The THREADPRIVATE directive and the common block or variable in it must appear in the same declaration section of a scoping unit
    !$omp threadprivate(i, j)
    i = 1
    j = 1
  end
end
