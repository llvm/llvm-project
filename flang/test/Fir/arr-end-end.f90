! Test affine pipeline
! RUN: bbc --emit-fir --gen-array-coor=true %s -o - | tco --fir-memref-dataflow-opt --fir-loop-result-opt --canonicalize  --loop-invariant-code-motion --promote-to-affine --affine-loop-invariant-code-motion --simplify-affine-structures --memref-dataflow-opt --cse --demote-affine --lower-affine | tco | llc | as -o %t
! RUN: %CC -std=c99 %t %S/arr-driver.c
! RUN: ./a.out | FileCheck %s

! CHECK: f1dc: success
subroutine f1dc(a1,a2,ret)
  integer a1(60), a2(60), ret(60)
  integer t1(60)

  do i = 1,60
     t1(i) = a1(i) + a1(i)
  end do
  do i = 1,60
     ret(i) = t1(i) * a2(i)
  end do
end subroutine f1dc

! CHECK: f1dv: success
subroutine f1dv(a1,a2,ret,s1)
  integer s1
  integer a1(s1), a2(s1), ret(s1)
  integer t1(s1)

  do i = 1,s1
     t1(i) = a1(i) + a1(i)
  end do
  do i = 1,s1
     ret(i) = t1(i) * a2(i)
  end do
end subroutine f1dv

! CHECK: f2dc: success
subroutine f2dc(a1,a2,ret)
  integer a1(3,3), a2(3,3), ret(3,3)
  integer t1(3,3)
  integer i,j

  do i = 1,3
     do j = 1,3
        t1(i,j) = a1(i,j) + a1(j,i)*100
     end do
  end do
  do i = 1,3
     do j = 1,3
        ret(i,j) = t1(i,j) + a2(i,j)*10000
     end do
  end do
end subroutine f2dc

! CHECK: f2dv: success
subroutine f2dv(a1,a2,ret,s1,s2)
  integer i,j,s1,s2
  integer a1(s1,s2), a2(s1,s2), ret(s1,s2)
  integer t(s1,s2)
  integer s1e,s2e
  s1e=s1
  s2e=s2
  do i = 1,s1e
     do j = 1,s2e
        t(i,j) = a1(i,j) + a1(j,i)*100
     end do
  end do
  do i = 1,s1e
     do j = 1,s2e
        ret(i,j) = t(i,j) + a2(i,j)*10000
     end do
  end do
end subroutine f2dv

! CHECK: f3dc: success
subroutine f3dc(a1,a2,ret)
  integer a1(3,3,3), a2(3,3,3), ret(3,3,3)
  integer t1(3,3,3)
  integer i,j,k

  do i = 1,3
     do j = 1,3
        do k = 1,3
           t1(i,j,k) = a1(i,j,k) + a1(k,j,i)*100
        end do
     end do
  end do
  do i = 1,3
     do j = 1,3
        do k = 1,3
           ret(i,j,k) = t1(i,j,k) + a2(i,j,k)*10000
        end do
     end do
  end do
end subroutine f3dc

! CHECK: f3dv: success
subroutine f3dv(a1,a2,ret,s1,s2,s3)
  integer i,j,k,s1,s2,s3
  integer a1(s1,s2,s3), a2(s1,s2,s3), ret(s1,s2,s3)
  integer t1(s1,s2,s3)
  integer s1e,s2e,s3e
  s1e=s1
  s2e=s2
  s3e=s3

  do i = 1,s1e
     do j = 1,s2e
        do k = 1,s3e
           t1(i,j,k) = a1(i,j,k) + a1(k,j,i)*100
        end do
     end do
  end do
  do i = 1,s1e
     do j = 1,s2e
        do k = 1,s3e
           ret(i,j,k) = t1(i,j,k) + a2(i,j,k)*10000
        end do
     end do
  end do
end subroutine f3dv
