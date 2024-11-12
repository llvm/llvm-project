!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %if x86-registered-target %{ %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fopenmp %s -o - | FileCheck --check-prefixes=CHECK,X86 %s %}
!RUN: %if aarch64-registerd-target %{ %flang_fc1 -triple aarch64-unknown-linux-gnu -emit-llvm -fopenmp %s -o - | FileCheck --check-prefixes=CHECK,AARCH64 %s %}

!CHECK: %[[X_NEW_VAL:.*]] = alloca { float, float }, align 8
!CHECK: %[[VAL_1:.*]] = alloca { float, float }, i64 1, align 8
!CHECK: %[[ORIG_VAL:.*]] = alloca { float, float }, i64 1, align 8
!CHECK: store { float, float } { float 2.000000e+00, float 2.000000e+00 }, ptr %[[ORIG_VAL]], align 4
!CHECK: br label %entry

!CHECK: entry:
!CHECK: %[[ATOMIC_TEMP_LOAD:.*]] = alloca { float, float }, align 8
!CHECK: call void @__atomic_load(i64 8, ptr %[[ORIG_VAL]], ptr %[[ATOMIC_TEMP_LOAD]], i32 0)
!CHECK: %[[PHI_NODE_ENTRY_1:.*]] = load { float, float }, ptr %[[ATOMIC_TEMP_LOAD]], align 8
!CHECK: br label %.atomic.cont

!CHECK: .atomic.cont
!CHECK: %[[VAL_4:.*]] = phi { float, float } [ %[[PHI_NODE_ENTRY_1]], %entry ], [ %{{.*}}, %.atomic.cont ]
!CHECK: %[[VAL_5:.*]] = extractvalue { float, float } %[[VAL_4]], 0
!CHECK: %[[VAL_6:.*]] = extractvalue { float, float } %[[VAL_4]], 1
!CHECK: %[[VAL_7:.*]] = fadd contract float %[[VAL_5]], 1.000000e+00
!CHECK: %[[VAL_8:.*]] = fadd contract float %[[VAL_6]], 1.000000e+00
!CHECK: %[[VAL_9:.*]] = insertvalue { float, float } undef, float %[[VAL_7]], 0
!CHECK: %[[VAL_10:.*]] = insertvalue { float, float } %[[VAL_9]], float %[[VAL_8]], 1
!CHECK: store { float, float } %[[VAL_10]], ptr %[[X_NEW_VAL]], align 4
!CHECK: %[[VAL_11:.*]] = call i1 @__atomic_compare_exchange(i64 8, ptr %[[ORIG_VAL]], ptr %[[ATOMIC_TEMP_LOAD]], ptr %[[X_NEW_VAL]],
!i32 2, i32 2)
!CHECK: %[[VAL_12:.*]] = load { float, float }, ptr %[[ATOMIC_TEMP_LOAD]], align 4
!CHECK: br i1 %[[VAL_11]], label %.atomic.exit, label %.atomic.cont

!CHECK:   .atomic.exit
!AARCH64: %[[LCSSA:.*]] = phi { float, float } [ %[[VAL_10]], %.atomic.cont ]
!AARCH64: store { float, float } %[[LCSSA]], ptr %[[VAL_1]], align 4
!X86:     store { float, float } %[[VAL_10]], ptr %[[VAL_1]], align 4

program main
      complex*8 ia, ib
      ia = (2, 2)
      !$omp atomic capture
        ia = ia + (1, 1)
        ib = ia
      !$omp end atomic
end program
