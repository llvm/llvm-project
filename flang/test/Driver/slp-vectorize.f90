! RUN: %flang -### -S -fslp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
! RUN: %flang -### -S -fno-slp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
! RUN: %flang -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
! RUN: %flang -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
! RUN: %flang -### -S -O2 %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
! RUN: %flang -### -S -O3 %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
! RUN: %flang -### -S -Os %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
! RUN: %flang -### -S -Oz %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
! RUN: %flang_fc1 -emit-llvm -O2 -vectorize-slp -mllvm -print-pipeline-passes -o /dev/null %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZER %s
! RUN: %flang_fc1 -emit-llvm -O2 -mllvm -print-pipeline-passes -o /dev/null %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZER %s
! CHECK-SLP-VECTORIZE: "-vectorize-slp"
! CHECK-NO-SLP-VECTORIZE-NOT: "-no-vectorize-slp"
! CHECK-SLP-VECTORIZER: slp-vectorizer
! CHECK-NO-SLP-VECTORIZER-NOT: slp-vectorizer

program test
end program
