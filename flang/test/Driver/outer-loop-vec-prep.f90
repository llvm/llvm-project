! RUN: %flang -### -O3 %s 2>&1 | FileCheck --check-prefix=O3 %s
! RUN: %flang -### -O2 %s 2>&1 | FileCheck --check-prefix=O2 %s
! RUN: %flang -### -O1 %s 2>&1 | FileCheck --check-prefix=O1 %s
! RUN: %flang -### %s 2>&1 | FileCheck --check-prefix=NOOPT %s

! O3: "-mllvm" "-enable-outer-loop-vectorization-prep"
! O2-NOT: enable-outer-loop-vectorization-prep
! O1-NOT: enable-outer-loop-vectorization-prep
! NOOPT-NOT: enable-outer-loop-vectorization-prep

subroutine test()
end subroutine
