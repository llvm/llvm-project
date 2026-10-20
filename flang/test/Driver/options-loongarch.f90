!! This test tests options from clang, which are also supported by flang in LoongArch.

! RUN: %flang -c --target=loongarch64-unknown-linux -mlsx %s -### 2>&1 | FileCheck --check-prefixes=LSX,NOLASX %s
! RUN: %flang -c --target=loongarch64-unknown-linux -mno-lsx %s -### 2>&1 | FileCheck --check-prefixes=NOLSX,NOLASX %s
! RUN: %flang -c --target=loongarch64-unknown-linux -mlasx %s -### 2>&1 | FileCheck --check-prefixes=LSX,LASX %s
! RUN: %flang -c --target=loongarch64-unknown-linux -mno-lasx %s -### 2>&1 | FileCheck --check-prefixes=LSX,NOLASX %s
! RUN: %flang -c --target=loongarch64-unknown-linux -msimd=none %s -### 2>&1 | FileCheck --check-prefixes=NOLSX,NOLASX %s
! RUN: %flang -c --target=loongarch64-unknown-linux -msimd=lsx %s -### 2>&1 | FileCheck --check-prefixes=LSX,NOLASX %s
! RUN: %flang -c --target=loongarch64-unknown-linux -msimd=lasx %s -### 2>&1 | FileCheck --check-prefixes=LSX,LASX %s
! RUN: not %flang -c --target=loongarch64-unknown-linux -msimd=supper %s -### 2>&1 | FileCheck --check-prefix=MSIMD-INVALID %s
! RUN: %flang -c --target=loongarch64-unknown-linux -mfrecipe %s -### 2>&1 | FileCheck --check-prefix=FRECIPE %s
! RUN: %flang -c --target=loongarch64-unknown-linux -mno-frecipe %s -### 2>&1 | FileCheck --check-prefix=NOFRECIPE %s
! RUN: %flang -c --target=loongarch64-unknown-linux -mlam-bh %s -### 2>&1 | FileCheck --check-prefix=LAMBH %s
! RUN: %flang -c --target=loongarch64-unknown-linux -mno-lam-bh %s -### 2>&1 | FileCheck --check-prefix=NOLAMBH %s
! RUN: %flang -c --target=loongarch64-unknown-linux -mlamcas %s -### 2>&1 | FileCheck --check-prefix=LAMCAS %s
! RUN: %flang -c --target=loongarch64-unknown-linux -mno-lamcas %s -### 2>&1 | FileCheck --check-prefix=NOLAMCAS %s
! RUN: %flang -c --target=loongarch64-unknown-linux -mld-seq-sa %s -### 2>&1 | FileCheck --check-prefix=LD-SEQ-SA %s
! RUN: %flang -c --target=loongarch64-unknown-linux -mno-ld-seq-sa %s -### 2>&1 | FileCheck --check-prefix=NOLD-SEQ-SA %s
! RUN: %flang -c --target=loongarch64-unknown-linux -mdiv32 %s -### 2>&1 | FileCheck --check-prefix=DIV32 %s
! RUN: %flang -c --target=loongarch64-unknown-linux -mno-div32 %s -### 2>&1 | FileCheck --check-prefix=NODIV32 %s
! RUN: %flang -c --target=loongarch64-unknown-linux -mannotate-tablejump %s -### 2>&1 | FileCheck --check-prefix=ANOTATE %s
! RUN: %flang -c --target=loongarch64-unknown-linux -mno-annotate-tablejump %s -### 2>&1 | FileCheck --check-prefix=NOANOTATE %s

! MSIMD-INVALID: error: invalid argument 'supper' to -msimd=; must be one of: none, lsx, lasx
! FRECIPE: "-target-feature" "+frecipe"
! NOFRECIPE-NOT: "-target-feature" "+frecipe"
! LAMBH: "-target-feature" "+lam-bh"
! NOLAMBH-NOT: "-target-feature" "+lam-bh"
! LAMCAS: "-target-feature" "+lamcas"
! NOLAMCAS-NOT: "-target-feature" "+lamcas"
! LD-SEQ-SA: "-target-feature" "+ld-seq-sa"
! NOLD-SEQ-SA-NOT: "-target-feature" "+ld-seq-sa"
! DIV32: "-target-feature" "+div32"
! NODIV32-NOT: "-target-feature" "+div32"
! ANOTATE: "-mllvm" "-loongarch-annotate-tablejump"
! NOANOTATE-NOT: "-loongarch-annotate-tablejump"

! NOLSX-NOT: "-target-feature" "+lsx"
! NOLASX-NOT: "-target-feature" "+lasx"
! LSX-DAG: "-target-feature" "+lsx"
! LASX-DAG: "-target-feature" "+lasx"
! NOLSX-NOT: "-target-feature" "+lsx"
! NOLASX-NOT: "-target-feature" "+lasx"
