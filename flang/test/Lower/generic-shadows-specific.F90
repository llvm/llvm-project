
#if STEP == 1
! these modules must be read from  module files
module generic_shadows_specific_m1
  interface f ! reference must be to generic
    module procedure f ! must have same name as generic interface
  end interface
 contains
  character function f() ! must be character
    f = 'q'
  end
end
module generic_shadows_specific_m2
   use generic_shadows_specific_m1
end
module generic_shadows_specific_m3
   use generic_shadows_specific_m2 ! must be generic_shadows_specific_m2, not generic_shadows_specific_m1
 contains
   subroutine mustExist() ! not called, but must exist
     character x
     x = f()
   end
end

#else
! Check that expected code produced with no crash.
subroutine reproducer()
  use generic_shadows_specific_m2
  use generic_shadows_specific_m3
  character x
  x = f()
end
#endif

!RUN: rm -rf %t && mkdir -p %t
!RUN: %flang_fc1 -fsyntax-only -DSTEP=1 -J%t %s
!RUN: %flang_fc1 -emit-fir -J%t -o - %s | FileCheck %s

!CHECK-LABEL: func.func @_QPreproducer
!CHECK: fir.call @_QMgeneric_shadows_specific_m1Pf
