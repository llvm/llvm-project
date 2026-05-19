! RUN: %python %S/../test_errors.py %s %flang -fopenmp -cpp -DNEGATIVE
! RUN: %flang_fc1 -fopenmp -fdebug-dump-symbols %s | FileCheck %s
! Test declare simd with uniform clause

function add2(a,b,i,fact,alc) result(c)
  !$omp declare simd(add2) uniform(a,b,fact)
  integer :: i
  integer,pointer::alc
  double precision :: a(*),b(*),fact,c
  c = a(i) + b(i) + fact
end function

end

! CHECK-LABEL: Subprogram scope: add2 size=48 alignment=8 sourceRange=189 bytes
! CHECK-NEXT:    a (OmpUniform): ObjectEntity dummy type: REAL(8) shape: 1_8:*
! CHECK-NEXT:    add2 (Function): HostAssoc
! CHECK-NEXT:    alc, POINTER size=24 offset=8: ObjectEntity dummy type: INTEGER(4)
! CHECK-NEXT:    b (OmpUniform): ObjectEntity dummy type: REAL(8) shape: 1_8:*
! CHECK-NEXT:    c size=8 offset=40: ObjectEntity funcResult type: REAL(8)
! CHECK-NEXT:    fact (OmpUniform) size=8 offset=32: ObjectEntity dummy type: REAL(8)
! CHECK-NEXT:    i size=4 offset=0: ObjectEntity dummy type: INTEGER(4)

#ifdef NEGATIVE

function bad_uniform(a,b,i) result(c)
  double precision :: a(*), b(*), c
  integer :: i
  double precision :: local
  !ERROR: Variable 'local' in UNIFORM clause must be a dummy argument of the enclosing procedure
  !$omp declare simd(bad_uniform) uniform(a,local)
  c = a(i) + b(i) + local
end function

#endif
