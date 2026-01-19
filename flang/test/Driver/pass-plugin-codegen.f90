! This test checks that the pre-codegen hook of LLVM pass plugins is executed
! before the code generation pipeline. The hook can also replace the output
! with its own.

! UNSUPPORTED: system-windows

! REQUIRES: plugins, examples
! Plugins are currently broken on AIX, at least in the CI.
! XFAIL: system-aix

! Without -last-words the pass does nothing, flang emits assembly.
! RUN: %flang_fc1 -S %s -o - %loadbye \
! RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-INACTIVE

! With -last-words the pass intercepts the code generation and prints
! "CodeGen Bye" instead.
! RUN: %flang_fc1 -S %s -o - %loadbye -mllvm -last-words \
! RUN: | FileCheck %s --check-prefix=CHECK-ACTIVE

! When emitting LLVM IR, no back-end is executed and therefore no
! pre-codegen hook of LLVM pass plugins are executed.
! RUN: %flang_fc1 -emit-llvm %s -o - %loadbye -mllvm -last-words \
! RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-LLVM

! Bye fails and reports an error when an object file should be emitted.
! (Note that this is specific for Bye; other plugins can support this.)
! RUN: not %flang_fc1 -emit-obj %s -o - %loadbye -mllvm -last-words \
! RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-ERR

! CHECK-INACTIVE-NOT: Bye
! CHECK-INACTIVE: empty_:
! CHECK-ACTIVE: CodeGen Bye
! CHECK-LLVM-NOT: Bye
! CHECK-LLVM: define{{.*}} void @empty_
! CHECK-ERR: error: last words unsupported for binary output

subroutine empty
end subroutine empty
