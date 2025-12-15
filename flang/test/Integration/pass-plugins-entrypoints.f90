! REQUIRES: plugins, examples

! Entry-points in default and -O0 pipeline
!
! RUN: %flang -fpass-plugin=%llvmshlibdir/Bye%pluginext \
! RUN:        -Xflang -load -Xflang %llvmshlibdir/Bye%pluginext \
! RUN:        -mllvm -print-ep-callbacks -o /dev/null -S %s | FileCheck --check-prefix=EP %s
!
! RUN: %flang -fpass-plugin=%llvmshlibdir/Bye%pluginext -flto=full -O0 \
! RUN:        -Xflang -load -Xflang %llvmshlibdir/Bye%pluginext \
! RUN:        -mllvm -print-ep-callbacks -o /dev/null -S %s | FileCheck --check-prefix=EP %s
!
! RUN: %flang -fpass-plugin=%llvmshlibdir/Bye%pluginext -flto=thin -O0 \
! RUN:        -Xflang -load -Xflang %llvmshlibdir/Bye%pluginext \
! RUN:        -mllvm -print-ep-callbacks -o /dev/null -S %s | FileCheck --check-prefix=EP %s
!
! EP:     PipelineStart
! EP:     PipelineEarlySimplification
! EP-NOT: Peephole
! EP:     ScalarOptimizerLate
! EP-NOT: Peephole
! EP:     OptimizerEarly
! EP:     VectorizerStart
! EP:     VectorizerEnd
! EP:     OptimizerLast

! Entry-points in optimizer pipeline
!
! RUN: %flang -fpass-plugin=%llvmshlibdir/Bye%pluginext -O2 \
! RUN:        -Xflang -load -Xflang %llvmshlibdir/Bye%pluginext \
! RUN:        -mllvm -print-ep-callbacks -o /dev/null -S %s | FileCheck --check-prefix=EP-OPT %s
!
! RUN: %flang -fpass-plugin=%llvmshlibdir/Bye%pluginext -O2 -flto=full \
! RUN:        -Xflang -load -Xflang %llvmshlibdir/Bye%pluginext \
! RUN:        -mllvm -print-ep-callbacks -o /dev/null -S %s | FileCheck --check-prefix=EP-OPT %s
!
! EP-OPT: PipelineStart
! EP-OPT: PipelineEarlySimplification
! EP-OPT: Peephole
! EP-OPT: ScalarOptimizerLate
! EP-OPT: Peephole
! EP-OPT: OptimizerEarly
! EP-OPT: VectorizerStart
! EP-OPT: VectorizerEnd
! EP-OPT: OptimizerLast

! FIXME: Thin-LTO does not invoke vectorizer callbacks
!
! RUN: %flang -fpass-plugin=%llvmshlibdir/Bye%pluginext -O2 -flto=thin \
! RUN:        -Xflang -load -Xflang %llvmshlibdir/Bye%pluginext \
! RUN:        -mllvm -print-ep-callbacks -o /dev/null -S %s | FileCheck --check-prefix=EP-LTO-THIN %s
!
! EP-LTO-THIN:     PipelineStart
! EP-LTO-THIN:     PipelineEarlySimplification
! EP-LTO-THIN:     Peephole
! EP-LTO-THIN:     ScalarOptimizerLate
! EP-LTO-THIN:     OptimizerEarly
! EP-LTO-THIN-NOT: Vectorizer
! EP-LTO-THIN:     OptimizerLast

INTEGER FUNCTION f(x)
  INTEGER, INTENT(IN) :: x
  f = x
END FUNCTION f
