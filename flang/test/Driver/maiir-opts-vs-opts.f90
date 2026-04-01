! Verify that aiir pass options are only accessible under `-maiir`.

!RUN: %flang_fc1 -emit-hlfir -maiir -aiir-pass-statistics %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=MAIIR
!RUN: not %flang_fc1 -emit-hlfir -aiir-pass-statistics %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOMAIIR

!MAIIR: Pass statistics report
!NOMAIIR: error: unknown argument: '-aiir-pass-statistics'
end
