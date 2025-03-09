!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %if x86-registered-target %{ %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fopenmp -mllvm --disable-llvm %s -o - | FileCheck %s %}
!RUN: %if aarch64-registerd-target %{ %flang_fc1 -triple aarch64-unknown-linux-gnu -emit-llvm -fopenmp -mllvm --disable-llvm %s -o - | FileCheck %s %}

! CHECK-LABEL: define {{.*}}@_QQmain(
! CHECK-NEXT:    %[[DOTATOMIC_ORIG_PTR:.+]] = alloca { float, float }, align 8
! CHECK-NEXT:    %[[DOTATOMIC_UPD_PTR:.+]] = alloca { float, float }, align 8
! CHECK-NEXT:    %[[TMP1:.+]] = alloca { float, float }, i64 1, align 8
! CHECK-NEXT:    %[[TMP2:.+]] = alloca { float, float }, i64 1, align 8
! CHECK-NEXT:    store { float, float } { float 2.000000e+00, float 2.000000e+00 }, ptr %[[TMP2]], align 4
! CHECK-NEXT:    br label %[[ENTRY:.+]]
! CHECK-EMPTY:
! CHECK-NEXT:  [[ENTRY]]:
! CHECK-NEXT:    %[[DOTATOMIC_LOAD:.+]] = load atomic i64, ptr %[[TMP2]] monotonic, align 8
! CHECK-NEXT:    store i64 %[[DOTATOMIC_LOAD]], ptr %[[DOTATOMIC_ORIG_PTR]], align 8
! CHECK-NEXT:    br label %[[DOTATOMIC_RETRY:.+]]
! CHECK-EMPTY:
! CHECK-NEXT:  [[DOTATOMIC_RETRY]]:
! CHECK-NEXT:    %[[DOTATOMIC_ORIG:.+]] = load { float, float }, ptr %[[DOTATOMIC_ORIG_PTR]], align 4
! CHECK-NEXT:    %[[TMP3:.+]] = extractvalue { float, float } %[[DOTATOMIC_ORIG]], 0
! CHECK-NEXT:    %[[TMP4:.+]] = extractvalue { float, float } %[[DOTATOMIC_ORIG]], 1
! CHECK-NEXT:    %[[TMP5:.+]] = fadd contract float %[[TMP3]], 1.000000e+00
! CHECK-NEXT:    %[[TMP6:.+]] = fadd contract float %[[TMP4]], 1.000000e+00
! CHECK-NEXT:    %[[TMP7:.+]] = insertvalue { float, float } undef, float %[[TMP5]], 0
! CHECK-NEXT:    %[[TMP8:.+]] = insertvalue { float, float } %[[TMP7]], float %[[TMP6]], 1
! CHECK-NEXT:    store { float, float } %[[TMP8]], ptr %[[DOTATOMIC_UPD_PTR]], align 4
! CHECK-NEXT:    %[[DOTCMPXCHG_EXPECTED:.+]] = load i64, ptr %[[DOTATOMIC_ORIG_PTR]], align 8
! CHECK-NEXT:    %[[DOTCMPXCHG_DESIRED:.+]] = load i64, ptr %[[DOTATOMIC_UPD_PTR]], align 8
! CHECK-NEXT:    %[[DOTCMPXCHG_PAIR:.+]] = cmpxchg weak ptr %[[TMP2]], i64 %[[DOTCMPXCHG_EXPECTED]], i64 %[[DOTCMPXCHG_DESIRED]] monotonic monotonic, align 8
! CHECK-NEXT:    %[[DOTCMPXCHG_PREV:.+]] = extractvalue { i64, i1 } %[[DOTCMPXCHG_PAIR]], 0
! CHECK-NEXT:    store i64 %[[DOTCMPXCHG_PREV]], ptr %[[DOTATOMIC_ORIG_PTR]], align 8
! CHECK-NEXT:    %[[DOTCMPXCHG_SUCCESS:.+]] = extractvalue { i64, i1 } %[[DOTCMPXCHG_PAIR]], 1
! CHECK-NEXT:    br i1 %[[DOTCMPXCHG_SUCCESS]], label %[[DOTATOMIC_DONE:.+]], label %[[DOTATOMIC_RETRY]]
! CHECK-EMPTY:
! CHECK-NEXT:  [[DOTATOMIC_DONE]]:
! CHECK-NEXT:    store { float, float } %[[TMP8]], ptr %[[TMP1]], align 4
! CHECK-NEXT:    ret void
! CHECK-NEXT:  }

program main
      complex*8 ia, ib
      ia = (2, 2)
      !$omp atomic capture
        ia = ia + (1, 1)
        ib = ia
      !$omp end atomic
end program
