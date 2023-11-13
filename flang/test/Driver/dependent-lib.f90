
! RUN: %flang_fc1 -emit-mlir -triple aarch64-pc-windows-msvc --dependent-lib=libtest %s -o - 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-mlir -triple x86_64-pc-windows-msvc --dependent-lib=libtest %s -o - 2>&1 | FileCheck %s

! CHECK: llvm.linker_options ["/DEFAULTLIB:", "libtest"]
program test
end program test