! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: _QPsub

! F77 code for the array computation c = ((a + b) * c) + (b / 2.0).
! (Eventually, test that the temporary arrays are eliminated.)
subroutine sub(a,b,c,i,j,k)
  real a(i,j,k), b(i,j,k), c(i,j,k)
  real t1(i,j,k), t2(i,j,k)
  integer i, j, k
  integer r, s, t

  do t = 1, k
     do s = 1, j
        do r = 1, i
           t1(r,s,t) = a(r,s,t) + b(r,s,t)
        end do
     end do
  end do
  do t = 1, k
     do s = 1, j
        do r = 1, i
           t2(r,s,t) = t1(r,s,t) * c(r,s,t)
        end do
     end do
  end do
  do t = 1, k
     do s = 1, j
        do r = 1, i
           c(r,s,t) = t2(r,s,t) + b(r,s,t) / 2.0
        end do
     end do
  end do
end subroutine sub
