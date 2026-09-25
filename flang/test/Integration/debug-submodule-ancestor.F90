! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang_fc1 -fsyntax-only -DSTEP=1 -J%t %s
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -J%t %s -o - \
! RUN:   | FileCheck %s

! Test that compiling a submodule on its own leaves the ancestor a declaration,
! so that it still merges with the unit that defines it.

#if STEP == 1
module anc_shapes
  implicit none
  integer :: mod_var = 1
  interface
    module subroutine hello()
    end subroutine
  end interface
end module anc_shapes
#else
submodule (anc_shapes) impl
contains
  module subroutine hello()
  end subroutine hello
end submodule impl

subroutine standalone()
  use anc_shapes
  mod_var = 2
end subroutine standalone
#endif

! CHECK-DAG: !DIModule(scope: ![[#]], name: "anc_shapes.impl", file: ![[#]], line: 19)
! CHECK-DAG: !DIModule(scope: null, name: "anc_shapes", isDecl: true)
