! RUN: not %flang_fc1 -triple x86_64-apple-macos10.13 -flto -ffat-lto-objects -emit-llvm-bc %s 2>&1 | FileCheck %s --check-prefix=ERROR
! ERROR: error: unsupported option '-ffat-lto-objects' for target 'x86_64-apple-macos10.13'

parameter(i=1)
integer :: j
end program
