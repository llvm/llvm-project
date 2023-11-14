! DEFINE: %{triple} =
! DEFINE: %{run} = %flang_fc1 -emit-mlir -triple %{triple} --dependent-lib=libtest %s -o - 2>&1 | FileCheck %s
! REDEFINE: %{triple} = aarch64-pc-windows-msvc
! RUN: %{run}
! REDEFINE: %{triple} = x86_64-pc-windows-msvc
! RUN: %{run}
! REDEFINE: %{triple} = x86_64-linux-unknown-gnu
! RUN: not %{run} --check-prefixes=CHECK-NOWIN
! REDEFINE: %{triple} = aarch64-apple-darwin
! RUN: not %{run} --check-prefixes=CHECK-NOWIN

! CHECK: llvm.linker_options ["/DEFAULTLIB:", "libtest"]
program test
end program test
! CHECK-NOWIN: --dependent-lib is only supported on Windows
