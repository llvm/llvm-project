! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-LLVM
! REQUIRES: target=powerpc{{.*}}

! CHECK-LLVM-LABEL: define void @_QQmain
      program ppc_vec_unit
      implicit none

      ! CHECK-LLVM-DAG: %[[VI2:.*]] = alloca <4 x i32>, i64 1, align 16
      ! CHECK-LLVM-DAG: %[[VI1:.*]] = alloca <4 x i32>, i64 1, align 16
      vector(integer(4)) :: vi1, vi2

      ! CHECK-LLVM-DAG: %[[VR2:.*]] = alloca <2 x double>, i64 1, align 16
      ! CHECK-LLVM-DAG: %[[VR1:.*]] = alloca <2 x double>, i64 1, align 16
      vector(real(8)) :: vr1, vr2

      ! CHECK-LLVM-DAG: %[[VU2:.*]] = alloca <8 x i16>, i64 1, align 16
      ! CHECK-LLVM-DAG: %[[VU1:.*]] = alloca <8 x i16>, i64 1, align 16
      vector(unsigned(2)) :: vu1, vu2

      ! CHECK-LLVM-DAG: %[[VP2:.*]] = alloca <256 x i1>, i64 1, align 32
      ! CHECK-LLVM-DAG: %[[VP1:.*]] = alloca <256 x i1>, i64 1, align 32
      __vector_pair :: vp1, vp2

      __vector_quad :: vq1, vq2

      ! CHECK-LLVM: %[[RESI:.*]] = call <4 x i32> @_QFPtest_vec_integer_assign(ptr %[[VI1]])
      vi2 = test_vec_integer_assign(vi1)
      ! CHECK-LLVM-NEXT: store <4 x i32> %[[RESI]], ptr %[[VI2]], align 16

      ! CHECK-LLVM-NEXT: %[[RESR:.*]] = call {{.*}}<2 x double> @_QFPtest_vec_real_assign(ptr %[[VR1]])
      vr2 = test_vec_real_assign(vr1)
      ! CHECK-LLVM-NEXT: store <2 x double> %[[RESR]], ptr %[[VR2]], align 16

      ! CHECK-LLVM-NEXT: %[[RESU:.*]] = call <8 x i16> @_QFPtest_vec_unsigned_assign(ptr %[[VU1]])
      vu2 = test_vec_unsigned_assign(vu1)
      ! CHECK-LLVM-NEXT: store <8 x i16> %[[RESU]], ptr %[[VU2]], align 16

      ! CHECK-LLVM-NEXT: %[[RESP:.*]] = call <256 x i1> @_QFPtest_vec_pair_assign(ptr %[[VP1]])
      vp2 = test_vec_pair_assign(vp1)
      ! CHECK-LLVM-NEXT: store <256 x i1> %[[RESP]], ptr %[[VP2]], align 32

      ! CHECK-LLVM-NEXT: %[[RESQ:.*]] = call <512 x i1> @_QFPtest_vec_quad_assign(ptr @_QFEvq1)
      vq2 = test_vec_quad_assign(vq1)
      ! CHECK-LLVM-NEXT: store <512 x i1> %[[RESQ]], ptr @_QFEvq2, align 64

      contains
      ! CHECK-LLVM-LABEL: define internal <4 x i32> @_QFPtest_vec_integer_assign
      function test_vec_integer_assign(arg1)
        ! CHECK-LLVM: %[[FUNC_RES:.*]] = alloca <4 x i32>, i64 1, align 16
        vector(integer(4)) :: arg1, test_vec_integer_assign

        ! CHECK-LLVM-NEXT: %[[ARG0:.*]] = load <4 x i32>, ptr %0, align 16
        ! CHECK-LLVM-NEXT: store <4 x i32> %[[ARG0]], ptr %[[FUNC_RES]], align 16

        test_vec_integer_assign = arg1

        ! CHECK-LLVM-NEXT: %[[RET:.*]] = load <4 x i32>, ptr %[[FUNC_RES]], align 16
        ! CHECK-LLVM-NEXT: ret <4 x i32> %[[RET]]
      end function test_vec_integer_assign

      ! CHECK-LLVM-LABEL: define internal <2 x double> @_QFPtest_vec_real_assign
      function test_vec_real_assign(arg1)
        ! CHECK-LLVM: %[[FUNC_RES:.*]] = alloca <2 x double>, i64 1, align 16
        vector(real(8)) :: arg1, test_vec_real_assign

        ! CHECK-LLVM-NEXT: %[[ARG0:.*]] = load <2 x double>, ptr %0, align 16
        ! CHECK-LLVM-NEXT: store <2 x double> %[[ARG0]], ptr %[[FUNC_RES]], align 16

        test_vec_real_assign = arg1

        ! CHECK-LLVM-NEXT: %[[RET:.*]] = load <2 x double>, ptr %[[FUNC_RES]], align 16
        ! CHECK-LLVM-NEXT: ret <2 x double> %[[RET]]
      end function test_vec_real_assign

      ! CHECK-LLVM-LABEL: define internal <8 x i16> @_QFPtest_vec_unsigned_assign
      function test_vec_unsigned_assign(arg1)
        ! CHECK-LLVM: %[[FUNC_RES:.*]] = alloca <8 x i16>, i64 1, align 16
        vector(unsigned(2)) :: arg1, test_vec_unsigned_assign

        ! CHECK-LLVM-NEXT: %[[ARG0:.*]] = load <8 x i16>, ptr %0, align 16
        ! CHECK-LLVM-NEXT: store <8 x i16> %[[ARG0]], ptr %[[FUNC_RES]], align 16

        test_vec_unsigned_assign = arg1

        ! CHECK-LLVM-NEXT: %[[RET:.*]] = load <8 x i16>, ptr %[[FUNC_RES]], align 16
        ! CHECK-LLVM-NEXT: ret <8 x i16> %[[RET]]
      end function test_vec_unsigned_assign

      ! CHECK-LLVM-LABEL: define internal <256 x i1> @_QFPtest_vec_pair_assign
      function test_vec_pair_assign(arg1)
        ! CHECK-LLVM: %[[FUNC_RES:.*]] = alloca <256 x i1>, i64 1, align 32
        __vector_pair :: arg1, test_vec_pair_assign

        ! CHECK-LLVM-NEXT: %[[ARG0:.*]] = load <256 x i1>, ptr %0, align 32
        ! CHECK-LLVM-NEXT: store <256 x i1> %[[ARG0]], ptr %[[FUNC_RES]], align 32

        test_vec_pair_assign = arg1

        ! CHECK-LLVM-NEXT: %[[RET:.*]] = load <256 x i1>, ptr %[[FUNC_RES]], align 32
        ! CHECK-LLVM-NEXT: ret <256 x i1> %[[RET]]
      end function test_vec_pair_assign

      ! CHECK-LLVM-LABEL: define internal <512 x i1> @_QFPtest_vec_quad_assign
      function test_vec_quad_assign(arg1)
        ! CHECK-LLVM: %[[FUNC_RES:.*]] = alloca <512 x i1>, i64 1, align 64
        __vector_quad :: arg1, test_vec_quad_assign

        ! CHECK-LLVM-NEXT: %[[ARG0:.*]] = load <512 x i1>, ptr %0, align 64
        ! CHECK-LLVM-NEXT: store <512 x i1> %[[ARG0]], ptr %[[FUNC_RES]], align 64

        test_vec_quad_assign = arg1

        ! CHECK-LLVM-NEXT: %[[RET:.*]] = load <512 x i1>, ptr %[[FUNC_RES]], align 64
        ! CHECK-LLVM-NEXT: ret <512 x i1> %[[RET]]
      end function test_vec_quad_assign

      end
