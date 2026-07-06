!RUN: %flang_fc1 -fdebug-unparse %s | FileCheck %s

module pdt05
  type base(k1,k2)
    integer(1),kind :: k1
    integer(k1),kind :: k2
    integer(kind(int(k1,1)+int(k2,k1))) j
    integer(kind(int(k1,1)+int(k2,kind(k2)))) k
  end type
end

use pdt05
type(base(2,7)) x27
type(base(8,7)) x87
print *, 'x27%j', kind(x27%j)
print *, 'x27%k', kind(x27%k)
print *, 'x87%j', kind(x87%j)
print *, 'x87%k', kind(x87%k)
end

!CHECK: PRINT *, "x27%j", 2_4
!CHECK: PRINT *, "x27%k", 2_4
!CHECK: PRINT *, "x87%j", 8_4
!CHECK: PRINT *, "x87%k", 8_4
