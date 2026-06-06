!RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s
!CHECK: p1a, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s1 bindName:p1a
!CHECK: p1b, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s1 bindName:p1b
!CHECK: p1c, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s1 bindName:P1c
!CHECK: p2a, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s2 bindName:p2a
!CHECK: p2b, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s2 bindName:p2b
!CHECK: p2c, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s2 bindName:P2c
!CHECK: p3a, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s3 bindName:p3a
!CHECK: p3b, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s3 bindName:p3b
!CHECK: p3c, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s3 bindName:P3c
module m1
  procedure(s1) :: p1a
  procedure(s1), bind(c) :: p1b
  procedure(s1), bind(c,name='P1c') :: p1c
  procedure(s2) :: p2a
  procedure(s2), bind(c) :: p2b
  procedure(s2), bind(c,name='P2c') :: p2c
  procedure(s3) :: p3a
  procedure(s3), bind(c) :: p3b
  procedure(s3), bind(c,name='P3c') :: p3c
 contains
  subroutine s1() bind(c)
  end
  subroutine s2() bind(c,name='')
  end
  subroutine s3() bind(c,name='foo')
  end
end

!CHECK: p1a, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s1 bindName:p1a
!CHECK: p1b, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s1 bindName:p1b
!CHECK: p1c, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s1 bindName:P1c
!CHECK: p2a, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s2 bindName:p2a
!CHECK: p2b, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s2 bindName:p2b
!CHECK: p2c, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s2 bindName:P2c
!CHECK: p3a, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s3 bindName:p3a
!CHECK: p3b, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s3 bindName:p3b
!CHECK: p3c, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s3 bindName:P3c
module m2
  interface
    subroutine s1() bind(c)
    end
    subroutine s2() bind(c,name='')
    end
    subroutine s3() bind(c,name='foo')
    end
  end interface
  procedure(s1) :: p1a
  procedure(s1), bind(c) :: p1b
  procedure(s1), bind(c,name='P1c') :: p1c
  procedure(s2) :: p2a
  procedure(s2), bind(c) :: p2b
  procedure(s2), bind(c,name='P2c') :: p2c
  procedure(s3) :: p3a
  procedure(s3), bind(c) :: p3b
  procedure(s3), bind(c,name='P3c') :: p3c
end

!CHECK: p1a, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s1 bindName:p1a
!CHECK: p1b, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s1 bindName:p1b
!CHECK: p1c, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s1 bindName:P1c
!CHECK: p2a, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s2 bindName:p2a
!CHECK: p2b, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s2 bindName:p2b
!CHECK: p2c, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s2 bindName:P2c
!CHECK: p3a, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s3 bindName:p3a
!CHECK: p3b, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s3 bindName:p3b
!CHECK: p3c, BIND(C), EXTERNAL, PUBLIC (Subroutine): ProcEntity s3 bindName:P3c
module m3
  procedure(s1) :: p1a
  procedure(s1), bind(c) :: p1b
  procedure(s1), bind(c,name='P1c') :: p1c
  procedure(s2) :: p2a
  procedure(s2), bind(c) :: p2b
  procedure(s2), bind(c,name='P2c') :: p2c
  procedure(s3) :: p3a
  procedure(s3), bind(c) :: p3b
  procedure(s3), bind(c,name='P3c') :: p3c
  interface
    subroutine s1() bind(c)
    end
    subroutine s2() bind(c,name='')
    end
    subroutine s3() bind(c,name='foo')
    end
  end interface
end

!CHECK: cdef01, BIND(C), PUBLIC size=4 offset=0: ObjectEntity type: REAL(4) bindName:cDef01 CDEFINED
module m4
  real, bind(c, name='cDef01', cdefined) :: cdef01
end
