! Verify that mlir pass options are only accessible under `-mmlir`.

!RUN: %flang_fc1 -emit-hlfir -mmlir -mlir-pass-statistics %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=MMLIR
!RUN: not %flang_fc1 -emit-hlfir -mlir-pass-statistics %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOMMLIR

!MMLIR: Pass statistics report
!NOMMLIR: error: unknown argument: '-mlir-pass-statistics'
end
