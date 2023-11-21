! Check that -falias-analysis and -fno-alias-analysis work as expected
! See flang/test/Fir/tbaa-codegen.fir for a test that the output is correct

! RUN: %flang -c -emit-llvm -falias-analysis %s -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-AA --check-prefix=CHECK-ALL
! RUN: %flang -c -emit-llvm -falias-analysis -fno-alias-analysis %s -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-NOAA --check-prefix=CHECK-ALL
! RUN: %flang -c -emit-llvm %s -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-NOAA --check-prefix=CHECK-ALL

! RUN: %flang -fc1 -emit-llvm -falias-analysis %s -o - | FileCheck %s --check-prefix=CHECK-AA --check-prefix=CHECK-ALL
! RUN: %flang -fc1 -emit-llvm -falias-analysis -fno-alias-analysis %s -o - | FileCheck %s --check-prefix=CHECK-NOAA --check-prefix=CHECK-ALL
! RUN: %flang -fc1 -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-NOAA --check-prefix=CHECK-ALL

subroutine simple(a)
  integer, intent(inout) :: a(:)
  a(1) = a(2)
end subroutine
! CHECK-ALL-LABEL: define void @simple
! CHECK-ALL:       ret
! CHECK-ALL:     }

! CHECK-AA: ![[ROOT:.*]] = !{!"Flang function root _QPsimple"}
! CHECK-NOAA-NOT: ![[ROOT:.*]] = !{!"Flang function root _QPsimple"}
