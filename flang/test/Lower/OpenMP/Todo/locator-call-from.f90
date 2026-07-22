!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -o - %s 2>&1 | FileCheck %s

!CHECK: not yet implemented: Function call locators are not supported yet

subroutine f
  interface
    function p
      integer, pointer :: p
    end
  end interface
  !$omp target update from(p())
end
