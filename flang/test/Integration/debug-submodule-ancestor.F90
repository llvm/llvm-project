! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang_fc1 -fsyntax-only -DSTEP=1 -J%t %s
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -J%t %s -o - \
! RUN:   | FileCheck %s

! Compiling a submodule on its own defines that submodule, not its ancestor.
! The ancestor is only used here, so it has to stay a declaration: were it given
! a file, a line and a scope, it would not merge with the definition emitted by
! the unit that really compiles it.

#if STEP == 1
module shapes
  implicit none
  integer :: mod_var = 1
  interface
    module subroutine hello()
    end subroutine
  end interface
end module shapes
#else
submodule (shapes) impl
contains
  module subroutine hello()
  end subroutine hello
end submodule impl

subroutine standalone()
  use shapes
  mod_var = 2
end subroutine standalone
#endif

! CHECK-DAG: !DIModule(scope: ![[#]], name: "shapes.impl", file: ![[#]], line: 21)
! CHECK-DAG: !DIModule(scope: null, name: "shapes", isDecl: true)
