!REQUIRES: x86-registered-target
!REQUIRES: flang-supports-f128-math
!RUN: %flang_fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | FileCheck  %s

! Test ABI of indirect calls is properly implemented in the LLVM IR.

subroutine foo(func_ptr, z)
  interface
    complex(16) function func_ptr()
    end function
  end interface
  complex(16) :: z
  ! CHECK: call void %{{.*}}(ptr sret({ fp128, fp128 }) align 16 %{{.*}})
  z = func_ptr()
end subroutine
