! Verify that a module using two modules, each with a name that cannot be written
! in a USE only-list (an unnamed nested DEC structure, or an OpenMP declare
! reduction of an intrinsic operator), writes a module file that can be read back
! by a separate compilation.

! RUN: rm -rf %t && split-file %s %t && mkdir %t/hermetic %t/use
! RUN: %flang_fc1 -fopenmp -fsyntax-only -J%t %t/p1.f90
! RUN: %flang_fc1 -fopenmp -fsyntax-only -J%t %t/p2.f90
! RUN: %flang_fc1 -fopenmp -fsyntax-only -J%t %t/fp.f90
! RUN: %flang_fc1 -fopenmp -fsyntax-only -J%t %t/left.f90
! RUN: %flang_fc1 -fopenmp -fsyntax-only -J%t %t/right.f90
! RUN: %flang_fc1 -fopenmp -fsyntax-only -J%t %t/facade.f90
! RUN: FileCheck --check-prefix=FP --input-file=%t/mf93_fp.mod %s
! RUN: FileCheck --check-prefix=FACADE --input-file=%t/mf93_facade.mod %s
! RUN: %flang_fc1 -fopenmp -fsyntax-only -J%t %t/client.f90
! RUN: %flang_fc1 -fopenmp -fsyntax-only -fhermetic-module-files -J%t/hermetic %t/p1.f90
! RUN: %flang_fc1 -fopenmp -fsyntax-only -fhermetic-module-files -J%t/hermetic %t/p2.f90
! RUN: %flang_fc1 -fopenmp -fsyntax-only -fhermetic-module-files -J%t/hermetic %t/fp.f90
! RUN: %flang_fc1 -fopenmp -fsyntax-only -fhermetic-module-files -J%t/hermetic %t/left.f90
! RUN: %flang_fc1 -fopenmp -fsyntax-only -fhermetic-module-files -J%t/hermetic %t/right.f90
! RUN: %flang_fc1 -fopenmp -fsyntax-only -fhermetic-module-files -J%t/hermetic %t/facade.f90
! RUN: cp %t/hermetic/mf93_fp.mod %t/hermetic/mf93_facade.mod %t/use
! RUN: %flang_fc1 -fopenmp -fsyntax-only -J%t/use %t/client.f90

! FP-DAG: {{^}}use mf93_p1{{$}}
! FP-DAG: {{^}}use mf93_p2{{$}}
! FP-NOT: only:{{.*}}F18
! FACADE-DAG: {{^}}use mf93_left{{$}}
! FACADE-DAG: {{^}}use mf93_right{{$}}
! FACADE-NOT: only:{{.*}}op\.

!--- p1.f90
module mf93_p1
  structure /s1/
    structure g
      integer i
    end structure
  end structure
  record /s1/ v1
end module

!--- p2.f90
module mf93_p2
  structure /s2/
    structure h
      real r
    end structure
  end structure
  record /s2/ v2
end module

!--- fp.f90
module mf93_fp
  use mf93_p1
  use mf93_p2
end module

!--- left.f90
module mf93_left
  type lt
    integer v
  end type
  !$omp declare reduction(+:lt:omp_out%v=omp_out%v+omp_in%v)
  !$omp declare reduction(*:lt:omp_out%v=omp_out%v*omp_in%v)
end module

!--- right.f90
module mf93_right
  type rt
    integer v
  end type
  !$omp declare reduction(+:rt:omp_out%v=omp_out%v+omp_in%v)
end module

!--- facade.f90
module mf93_facade
  use mf93_left
  use mf93_right
end module

!--- client.f90
program client
  use mf93_fp
  use mf93_facade
end program
