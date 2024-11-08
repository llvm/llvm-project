! RUN: %flang_fc1 -flang-experimental-hlfir -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR","LLVMIR-LE" %s
! RUN: %flang_fc1 -flang-experimental-hlfir -triple powerpc64-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR","LLVMIR-BE" %s
! REQUIRES: target=powerpc{{.*}}

! CHECK-LABEL: vec_perm_test_i1
subroutine vec_perm_test_i1(arg1, arg2, arg3)
  vector(integer(1)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[barg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR-LE: %[[xor:.*]] = xor <16 x i8> %[[arg3]], splat (i8 -1)
! LLVMIR-LE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg2]], <4 x i32> %[[barg1]], <16 x i8> %[[xor]])
! LLVMIR-BE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg1]], <4 x i32> %[[barg2]], <16 x i8> %[[arg3]])
! LLVMIR: %[[bcall:.*]] = bitcast <4 x i32> %[[call]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[bcall]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_i1

! CHECK-LABEL: vec_perm_test_i2
subroutine vec_perm_test_i2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR-LE: %[[xor:.*]] = xor <16 x i8> %[[arg3]], splat (i8 -1)
! LLVMIR-LE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg2]], <4 x i32> %[[barg1]], <16 x i8> %[[xor]])
! LLVMIR-BE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg1]], <4 x i32> %[[barg2]], <16 x i8> %[[arg3]])
! LLVMIR: %[[bcall:.*]] = bitcast <4 x i32> %[[call]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[bcall]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_i2

! CHECK-LABEL: vec_perm_test_i4
subroutine vec_perm_test_i4(arg1, arg2, arg3)
  vector(integer(4)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR-LE: %[[xor:.*]] = xor <16 x i8> %[[arg3]], splat (i8 -1)
! LLVMIR-LE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[arg2]], <4 x i32> %[[arg1]], <16 x i8> %[[xor]])
! LLVMIR-BE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]], <16 x i8> %[[arg3]])
! LLVMIR: store <4 x i32> %[[call]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_i4

! CHECK-LABEL: vec_perm_test_i8
subroutine vec_perm_test_i8(arg1, arg2, arg3)
  vector(integer(8)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <4 x i32>
! LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <4 x i32>
! LLVMIR-LE: %[[xor:.*]] = xor <16 x i8> %[[arg3]], splat (i8 -1)
! LLVMIR-LE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg2]], <4 x i32> %[[barg1]], <16 x i8> %[[xor]])
! LLVMIR-BE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg1]], <4 x i32> %[[barg2]], <16 x i8> %[[arg3]])
! LLVMIR: %[[bcall:.*]] = bitcast <4 x i32> %[[call]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[bcall]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_i8

! CHECK-LABEL: vec_perm_test_u1
subroutine vec_perm_test_u1(arg1, arg2, arg3)
  vector(unsigned(1)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[barg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR-LE: %[[xor:.*]] = xor <16 x i8> %[[arg3]], splat (i8 -1)
! LLVMIR-LE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg2]], <4 x i32> %[[barg1]], <16 x i8> %[[xor]])
! LLVMIR-BE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg1]], <4 x i32> %[[barg2]], <16 x i8> %[[arg3]])
! LLVMIR: %[[bcall:.*]] = bitcast <4 x i32> %[[call]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[bcall]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_u1

! CHECK-LABEL: vec_perm_test_u2
subroutine vec_perm_test_u2(arg1, arg2, arg3)
  vector(unsigned(2)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! LLVMIR-LE: %[[xor:.*]] = xor <16 x i8> %[[arg3]], splat (i8 -1)
! LLVMIR-LE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg2]], <4 x i32> %[[barg1]], <16 x i8> %[[xor]])
! LLVMIR-BE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg1]], <4 x i32> %[[barg2]], <16 x i8> %[[arg3]])
! LLVMIR: %[[bcall:.*]] = bitcast <4 x i32> %[[call]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[bcall]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_u2

! CHECK-LABEL: vec_perm_test_u4
subroutine vec_perm_test_u4(arg1, arg2, arg3)
  vector(unsigned(4)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR-LE: %[[xor:.*]] = xor <16 x i8> %[[arg3]], splat (i8 -1)
! LLVMIR-LE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[arg2]], <4 x i32> %[[arg1]], <16 x i8> %[[xor]])
! LLVMIR-BE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]], <16 x i8> %[[arg3]])
! LLVMIR: store <4 x i32> %[[call]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_u4

! CHECK-LABEL: vec_perm_test_u8
subroutine vec_perm_test_u8(arg1, arg2, arg3)
  vector(unsigned(8)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <4 x i32>
! LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <4 x i32>
! LLVMIR-LE: %[[xor:.*]] = xor <16 x i8> %[[arg3]], splat (i8 -1)
! LLVMIR-LE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg2]], <4 x i32> %[[barg1]], <16 x i8> %[[xor]])
! LLVMIR-BE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg1]], <4 x i32> %[[barg2]], <16 x i8> %[[arg3]])
! LLVMIR: %[[bcall:.*]] = bitcast <4 x i32> %[[call]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[bcall]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_u8

! CHECK-LABEL: vec_perm_test_r4
subroutine vec_perm_test_r4(arg1, arg2, arg3)
  vector(real(4)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <4 x i32>
! LLVMIR-LE: %[[xor:.*]] = xor <16 x i8> %[[arg3]], splat (i8 -1)
! LLVMIR-LE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg2]], <4 x i32> %[[barg1]], <16 x i8> %[[xor]])
! LLVMIR-BE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg1]], <4 x i32> %[[barg2]], <16 x i8> %[[arg3]])
! LLVMIR: %[[bcall:.*]] = bitcast <4 x i32> %[[call]] to <4 x float>
! LLVMIR: store <4 x float> %[[bcall]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_r4

! CHECK-LABEL: vec_perm_test_r8
subroutine vec_perm_test_r8(arg1, arg2, arg3)
  vector(real(8)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <4 x i32>
! LLVMIR: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <4 x i32>
! LLVMIR-LE: %[[xor:.*]] = xor <16 x i8> %[[arg3]], splat (i8 -1)
! LLVMIR-LE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg2]], <4 x i32> %[[barg1]], <16 x i8> %[[xor]])
! LLVMIR-BE: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg1]], <4 x i32> %[[barg2]], <16 x i8> %[[arg3]])
! LLVMIR: %[[bcall:.*]] = bitcast <4 x i32> %[[call]] to <2 x double>
! LLVMIR: store <2 x double> %[[bcall]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_r8

! CHECK-LABEL: vec_permi_test_i8i1
subroutine vec_permi_test_i8i1(arg1, arg2, arg3)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 1, i32 3>
! LLVMIR: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_i8i1

! CHECK-LABEL: vec_permi_test_i8i2
subroutine vec_permi_test_i8i2(arg1, arg2, arg3)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 2_2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 1, i32 2>
! LLVMIR: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_i8i2

! CHECK-LABEL: vec_permi_test_i8i4
subroutine vec_permi_test_i8i4(arg1, arg2, arg3)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 1_4)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 0, i32 3>
! LLVMIR: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_i8i4

! CHECK-LABEL: vec_permi_test_i8i8
subroutine vec_permi_test_i8i8(arg1, arg2, arg3)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 0_8)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 0, i32 2>
! LLVMIR: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_i8i8

! CHECK-LABEL: vec_permi_test_u8i1
subroutine vec_permi_test_u8i1(arg1, arg2, arg3)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 1, i32 3>
! LLVMIR: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_u8i1

! CHECK-LABEL: vec_permi_test_u8i2
subroutine vec_permi_test_u8i2(arg1, arg2, arg3)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 2_2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 1, i32 2>
! LLVMIR: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_u8i2

! CHECK-LABEL: vec_permi_test_u8i4
subroutine vec_permi_test_u8i4(arg1, arg2, arg3)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 1_4)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 0, i32 3>
! LLVMIR: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_u8i4

! CHECK-LABEL: vec_permi_test_u8i8
subroutine vec_permi_test_u8i8(arg1, arg2, arg3)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 0_8)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 0, i32 2>
! LLVMIR: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_u8i8

! CHECK-LABEL: vec_permi_test_r4i1
subroutine vec_permi_test_r4i1(arg1, arg2, arg3)
  vector(real(4)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <2 x double>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <2 x double>
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x double> %[[barg1]], <2 x double> %[[barg2]], <2 x i32> <i32 1, i32 3>
! LLVMIR: %[[bshuf:.*]] = bitcast <2 x double> %[[shuf]] to <4 x float>
! LLVMIR: store <4 x float> %[[bshuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_r4i1

! CHECK-LABEL: vec_permi_test_r4i2
subroutine vec_permi_test_r4i2(arg1, arg2, arg3)
  vector(real(4)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 2_2)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <2 x double>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <2 x double>
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x double> %[[barg1]], <2 x double> %[[barg2]], <2 x i32> <i32 1, i32 2>
! LLVMIR: %[[bshuf:.*]] = bitcast <2 x double> %[[shuf]] to <4 x float>
! LLVMIR: store <4 x float> %[[bshuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_r4i2

! CHECK-LABEL: vec_permi_test_r4i4
subroutine vec_permi_test_r4i4(arg1, arg2, arg3)
  vector(real(4)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 1_4)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <2 x double>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <2 x double>
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x double> %[[barg1]], <2 x double> %[[barg2]], <2 x i32> <i32 0, i32 3>
! LLVMIR: %[[bshuf:.*]] = bitcast <2 x double> %[[shuf]] to <4 x float>
! LLVMIR: store <4 x float> %[[bshuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_r4i4

! CHECK-LABEL: vec_permi_test_r4i8
subroutine vec_permi_test_r4i8(arg1, arg2, arg3)
  vector(real(4)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 0_8)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <2 x double>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <2 x double>
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x double> %[[barg1]], <2 x double> %[[barg2]], <2 x i32> <i32 0, i32 2>
! LLVMIR: %[[bshuf:.*]] = bitcast <2 x double> %[[shuf]] to <4 x float>
! LLVMIR: store <4 x float> %[[bshuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_r4i8

! CHECK-LABEL: vec_permi_test_r8i1
subroutine vec_permi_test_r8i1(arg1, arg2, arg3)
  vector(real(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x double> %[[arg1]], <2 x double> %[[arg2]], <2 x i32> <i32 1, i32 3>
! LLVMIR: store <2 x double> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_r8i1

! CHECK-LABEL: vec_permi_test_r8i2
subroutine vec_permi_test_r8i2(arg1, arg2, arg3)
  vector(real(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 2_2)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x double> %[[arg1]], <2 x double> %[[arg2]], <2 x i32> <i32 1, i32 2>
! LLVMIR: store <2 x double> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_r8i2

! CHECK-LABEL: vec_permi_test_r8i4
subroutine vec_permi_test_r8i4(arg1, arg2, arg3)
  vector(real(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 1_4)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x double> %[[arg1]], <2 x double> %[[arg2]], <2 x i32> <i32 0, i32 3>
! LLVMIR: store <2 x double> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_r8i4

! CHECK-LABEL: vec_permi_test_r8i8
subroutine vec_permi_test_r8i8(arg1, arg2, arg3)
  vector(real(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 0_8)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x double> %[[arg1]], <2 x double> %[[arg2]], <2 x i32> <i32 0, i32 2>
! LLVMIR: store <2 x double> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_r8i8
