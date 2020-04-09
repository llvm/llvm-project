! Tests for control-flow

! RUN: bbc -emit-fir %s -o - | FileCheck %s

! check the lowering of a RETURN in the body of a SUBROUTINE
! CHECK-LABEL one
subroutine one(a,b,c)
  d = 1.0
  if (a .ne. b) then
    ! CHECK: call @_QPone_a
    call one_a(d)
    ! CHECK: cond_br %{{.*}}, ^bb[[TB:.*]], ^
    if (d .eq. 1.0) then
       ! CHECK-NEXT: ^bb[[TB]]:
       ! CHECK-NEXT: br ^bb[[EXIT:.*]]
       return
    endif
 else
    e = 4.0
    call one_b(c,d,e)
 endif
 ! CHECK: ^bb[[EXIT]]:
 ! CHECK-NEXT: return
end subroutine one
    
