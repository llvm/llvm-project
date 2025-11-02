! Test lowering of atomic write to LLVM IR for complex types.
! This is a regression test for issue #165184.

! RUN: %flang_fc1 -emit-llvm -fopenmp -o - %s | FileCheck %s

! Test that atomic write operations with complex types emit the correct
! size parameter to __atomic_store:
! - complex(4) (8 bytes total): should call __atomic_store(i64 8, ...)
! - complex(8) (16 bytes total): should call __atomic_store(i64 16, ...)

program atomic_write_complex
  implicit none

  ! Test complex(4) - single precision (8 bytes)
  complex(4) :: c41, c42
  ! Test complex(8) - double precision (16 bytes)  
  complex(8) :: c81, c82
  
  c42 = (1.0_4, 1.0_4)
  c82 = (1.0_8, 1.0_8)

  ! CHECK-LABEL: define {{.*}} @_QQmain
  
  ! Single precision complex: 8 bytes
  ! CHECK: call void @__atomic_store(i64 8, ptr {{.*}}, ptr {{.*}}, i32 {{.*}})
!$omp atomic write
  c41 = c42
  
  ! Double precision complex: 16 bytes (this was broken before the fix)
  ! CHECK: call void @__atomic_store(i64 16, ptr {{.*}}, ptr {{.*}}, i32 {{.*}})
!$omp atomic write
  c81 = c82

end program atomic_write_complex
