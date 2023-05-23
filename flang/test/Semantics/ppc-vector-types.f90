! RUN: %flang_fc1 -fdebug-unparse %s | FileCheck %s
! REQUIRES: target=powerpc{{.*}}

    ! CHECK-LABEL: PROGRAM ppc_vec_unit
      program ppc_vec_unit
      implicit none
      ! CHECK: VECTOR(INTEGER(KIND=4_4)) :: vi1, vi2
      vector(integer(4)) :: vi1, vi2
      ! CHECK-NEXT: VECTOR(REAL(KIND=8_4)) :: vr1, vr2
      vector(real(8)) :: vr1, vr2
      ! CHECK-NEXT: VECTOR(UNSIGNED(KIND=2_4)) :: vu1, vu2
      vector(unsigned(2)) :: vu1, vu2
      ! CHECK-NEXT: __VECTOR_PAIR :: vp1, vp2
      __vector_pair :: vp1, vp2
      ! CHECK-NEXT: __VECTOR_QUAD :: vq1, vq2
      __vector_quad :: vq1, vq2
      ! CHECK-NEXT: vi2=test_vec_integer_assign(vi1)
      vi2 = test_vec_integer_assign(vi1)
      ! CHECK-NEXT: vr2=test_vec_real_assign(vr1)
      vr2 = test_vec_real_assign(vr1)
      ! CHECK-NEXT: vu2=test_vec_unsigned_assign(vu1)
      vu2 = test_vec_unsigned_assign(vu1)
      ! CHECK-NEXT: vp2=test_vec_pair_assign(vp1)
      vp2 = test_vec_pair_assign(vp1)
      ! CHECK-NEXT: vq2=test_vec_quad_assign(vq1)
      vq2 = test_vec_quad_assign(vq1)

      contains
      ! CHECK-LABEL: FUNCTION test_vec_integer_assign
      function test_vec_integer_assign(arg1)
        ! CHECK: VECTOR(INTEGER(KIND=4_4)) :: arg1, test_vec_integer_assign
        vector(integer(4)) :: arg1, test_vec_integer_assign
        ! CHECK-NEXT: test_vec_integer_assign=arg1
        test_vec_integer_assign = arg1
      end function test_vec_integer_assign

      ! CHECK-LABEL: FUNCTION test_vec_real_assign
      function test_vec_real_assign(arg1)
        ! CHECK: VECTOR(REAL(KIND=8_4)) :: arg1, test_vec_real_assign
        vector(real(8)) :: arg1, test_vec_real_assign
        ! CHECK-NEXT: test_vec_real_assign=arg1
        test_vec_real_assign = arg1
      end function test_vec_real_assign

      ! CHECK-LABEL: FUNCTION test_vec_unsigned_assign
      function test_vec_unsigned_assign(arg1)
        ! CHECK: VECTOR(UNSIGNED(KIND=2_4)) :: arg1, test_vec_unsigned_assign
        vector(unsigned(2)) :: arg1, test_vec_unsigned_assign
        ! CHECK-NEXT: test_vec_unsigned_assign=arg1
        test_vec_unsigned_assign = arg1
      end function test_vec_unsigned_assign

      ! CHECK-LABEL: FUNCTION test_vec_pair_assign
      function test_vec_pair_assign(arg1)
        ! CHECK: __VECTOR_PAIR :: arg1, test_vec_pair_assign
        __vector_pair :: arg1, test_vec_pair_assign
        ! CHECK-NEXT: test_vec_pair_assign=arg1
        test_vec_pair_assign = arg1
      end function test_vec_pair_assign

      ! CHECK-LABEL: FUNCTION test_vec_quad_assign
      function test_vec_quad_assign(arg1)
        ! CHECK: __VECTOR_QUAD :: arg1, test_vec_quad_assign
        __vector_quad :: arg1, test_vec_quad_assign
        ! CHECK-NEXT: test_vec_quad_assign=arg1
        test_vec_quad_assign = arg1
      end function test_vec_quad_assign

      end
