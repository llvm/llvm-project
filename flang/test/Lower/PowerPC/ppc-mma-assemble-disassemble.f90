! RUN: %flang_fc1 -flang-experimental-hlfir -triple powerpc64le-unknown-unknown -target-cpu pwr10 -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

! mma_assemble_acc

      subroutine test_assemble_acc_i1()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_i1

! CHECK-LABEL: @test_assemble_acc_i1

! LLVMIR:         %[[VAL_0:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_1:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_2:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_3:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_4:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_5:.*]] = load <16 x i8>, ptr %[[VAL_3]], align 16
! LLVMIR:         %[[VAL_6:.*]] = load <16 x i8>, ptr %[[VAL_2]], align 16
! LLVMIR:         %[[VAL_7:.*]] = load <16 x i8>, ptr %[[VAL_1]], align 16
! LLVMIR:         %[[VAL_8:.*]] = load <16 x i8>, ptr %[[VAL_0]], align 16
! LLVMIR:         %[[VAL_9:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_5]], <16 x i8> %[[VAL_6]], <16 x i8> %[[VAL_7]], <16 x i8> %[[VAL_8]])
! LLVMIR:         store <512 x i1> %[[VAL_9]], ptr %[[VAL_4]], align 64

      subroutine test_assemble_acc_i2()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_i2

! CHECK-LABEL: @test_assemble_acc_i2
! LLVMIR:         %[[VAL_10:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_11:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_12:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_13:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_14:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_15:.*]] = load <8 x i16>, ptr %[[VAL_13]], align 16
! LLVMIR:         %[[VAL_16:.*]] = load <8 x i16>, ptr %[[VAL_12]], align 16
! LLVMIR:         %[[VAL_17:.*]] = load <8 x i16>, ptr %[[VAL_11]], align 16
! LLVMIR:         %[[VAL_18:.*]] = load <8 x i16>, ptr %[[VAL_10]], align 16
! LLVMIR:         %[[VAL_19:.*]] = bitcast <8 x i16> %[[VAL_15]] to <16 x i8>
! LLVMIR:         %[[VAL_20:.*]] = bitcast <8 x i16> %[[VAL_16]] to <16 x i8>
! LLVMIR:         %[[VAL_21:.*]] = bitcast <8 x i16> %[[VAL_17]] to <16 x i8>
! LLVMIR:         %[[VAL_22:.*]] = bitcast <8 x i16> %[[VAL_18]] to <16 x i8>
! LLVMIR:         %[[VAL_23:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_19]], <16 x i8> %[[VAL_20]], <16 x i8> %[[VAL_21]], <16 x i8> %[[VAL_22]])
! LLVMIR:         store <512 x i1> %[[VAL_23]], ptr %[[VAL_14]], align 64


      subroutine test_assemble_acc_i4()
      use, intrinsic :: mma
      implicit none
      vector(integer(4)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_i4

! CHECK-LABEL: @test_assemble_acc_i4
! LLVMIR:         %[[VAL_24:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_25:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_26:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_27:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_28:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_29:.*]] = load <4 x i32>, ptr %[[VAL_27]], align 16
! LLVMIR:         %[[VAL_30:.*]] = load <4 x i32>, ptr %[[VAL_26]], align 16
! LLVMIR:         %[[VAL_31:.*]] = load <4 x i32>, ptr %[[VAL_25]], align 16
! LLVMIR:         %[[VAL_32:.*]] = load <4 x i32>, ptr %[[VAL_24]], align 16
! LLVMIR:         %[[VAL_33:.*]] = bitcast <4 x i32> %[[VAL_29]] to <16 x i8>
! LLVMIR:         %[[VAL_34:.*]] = bitcast <4 x i32> %[[VAL_30]] to <16 x i8>
! LLVMIR:         %[[VAL_35:.*]] = bitcast <4 x i32> %[[VAL_31]] to <16 x i8>
! LLVMIR:         %[[VAL_36:.*]] = bitcast <4 x i32> %[[VAL_32]] to <16 x i8>
! LLVMIR:         %[[VAL_37:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_33]], <16 x i8> %[[VAL_34]], <16 x i8> %[[VAL_35]], <16 x i8> %[[VAL_36]])
! LLVMIR:         store <512 x i1> %[[VAL_37]], ptr %[[VAL_28]], align 64

      subroutine test_assemble_acc_i8()
      use, intrinsic :: mma
      implicit none
      vector(integer(8)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_i8

! CHECK-LABEL: @test_assemble_acc_i8
! LLVMIR:         %[[VAL_38:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_39:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_40:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_41:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_42:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_43:.*]] = load <2 x i64>, ptr %[[VAL_41]], align 16
! LLVMIR:         %[[VAL_44:.*]] = load <2 x i64>, ptr %[[VAL_40]], align 16
! LLVMIR:         %[[VAL_45:.*]] = load <2 x i64>, ptr %[[VAL_39]], align 16
! LLVMIR:         %[[VAL_46:.*]] = load <2 x i64>, ptr %[[VAL_38]], align 16
! LLVMIR:         %[[VAL_47:.*]] = bitcast <2 x i64> %[[VAL_43]] to <16 x i8>
! LLVMIR:         %[[VAL_48:.*]] = bitcast <2 x i64> %[[VAL_44]] to <16 x i8>
! LLVMIR:         %[[VAL_49:.*]] = bitcast <2 x i64> %[[VAL_45]] to <16 x i8>
! LLVMIR:         %[[VAL_50:.*]] = bitcast <2 x i64> %[[VAL_46]] to <16 x i8>
! LLVMIR:         %[[VAL_51:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_47]], <16 x i8> %[[VAL_48]], <16 x i8> %[[VAL_49]], <16 x i8> %[[VAL_50]])
! LLVMIR:         store <512 x i1> %[[VAL_51]], ptr %[[VAL_42]], align 64


      subroutine test_assemble_acc_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_u1

! CHECK-LABEL: @test_assemble_acc_u1
! LLVMIR:         %[[VAL_52:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_53:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_54:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_55:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_56:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_57:.*]] = load <16 x i8>, ptr %[[VAL_55]], align 16
! LLVMIR:         %[[VAL_58:.*]] = load <16 x i8>, ptr %[[VAL_54]], align 16
! LLVMIR:         %[[VAL_59:.*]] = load <16 x i8>, ptr %[[VAL_53]], align 16
! LLVMIR:         %[[VAL_60:.*]] = load <16 x i8>, ptr %[[VAL_52]], align 16
! LLVMIR:         %[[VAL_61:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_57]], <16 x i8> %[[VAL_58]], <16 x i8> %[[VAL_59]], <16 x i8> %[[VAL_60]])
! LLVMIR:         store <512 x i1> %[[VAL_61]], ptr %[[VAL_56]], align 64

      subroutine test_assemble_acc_u2()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(2)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_u2

! CHECK-LABEL: @test_assemble_acc_u2
! LLVMIR:         %[[VAL_62:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_63:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_64:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_65:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_66:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_67:.*]] = load <8 x i16>, ptr %[[VAL_65]], align 16
! LLVMIR:         %[[VAL_68:.*]] = load <8 x i16>, ptr %[[VAL_64]], align 16
! LLVMIR:         %[[VAL_69:.*]] = load <8 x i16>, ptr %[[VAL_63]], align 16
! LLVMIR:         %[[VAL_70:.*]] = load <8 x i16>, ptr %[[VAL_62]], align 16
! LLVMIR:         %[[VAL_71:.*]] = bitcast <8 x i16> %[[VAL_67]] to <16 x i8>
! LLVMIR:         %[[VAL_72:.*]] = bitcast <8 x i16> %[[VAL_68]] to <16 x i8>
! LLVMIR:         %[[VAL_73:.*]] = bitcast <8 x i16> %[[VAL_69]] to <16 x i8>
! LLVMIR:         %[[VAL_74:.*]] = bitcast <8 x i16> %[[VAL_70]] to <16 x i8>
! LLVMIR:         %[[VAL_75:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_71]], <16 x i8> %[[VAL_72]], <16 x i8> %[[VAL_73]], <16 x i8> %[[VAL_74]])
! LLVMIR:         store <512 x i1> %[[VAL_75]], ptr %[[VAL_66]], align 64

      subroutine test_assemble_acc_u4()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(4)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_u4

! CHECK-LABEL: @test_assemble_acc_u4
! LLVMIR:         %[[VAL_76:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_77:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_78:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_79:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_80:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_81:.*]] = load <4 x i32>, ptr %[[VAL_79]], align 16
! LLVMIR:         %[[VAL_82:.*]] = load <4 x i32>, ptr %[[VAL_78]], align 16
! LLVMIR:         %[[VAL_83:.*]] = load <4 x i32>, ptr %[[VAL_77]], align 16
! LLVMIR:         %[[VAL_84:.*]] = load <4 x i32>, ptr %[[VAL_76]], align 16
! LLVMIR:         %[[VAL_85:.*]] = bitcast <4 x i32> %[[VAL_81]] to <16 x i8>
! LLVMIR:         %[[VAL_86:.*]] = bitcast <4 x i32> %[[VAL_82]] to <16 x i8>
! LLVMIR:         %[[VAL_87:.*]] = bitcast <4 x i32> %[[VAL_83]] to <16 x i8>
! LLVMIR:         %[[VAL_88:.*]] = bitcast <4 x i32> %[[VAL_84]] to <16 x i8>
! LLVMIR:         %[[VAL_89:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_85]], <16 x i8> %[[VAL_86]], <16 x i8> %[[VAL_87]], <16 x i8> %[[VAL_88]])
! LLVMIR:         store <512 x i1> %[[VAL_89]], ptr %[[VAL_80]], align 64

      subroutine test_assemble_acc_u8()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(8)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_u8

! CHECK-LABEL: @test_assemble_acc_u8
! LLVMIR:         %[[VAL_90:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_91:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_92:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_93:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_94:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_95:.*]] = load <2 x i64>, ptr %[[VAL_93]], align 16
! LLVMIR:         %[[VAL_96:.*]] = load <2 x i64>, ptr %[[VAL_92]], align 16
! LLVMIR:         %[[VAL_97:.*]] = load <2 x i64>, ptr %[[VAL_91]], align 16
! LLVMIR:         %[[VAL_98:.*]] = load <2 x i64>, ptr %[[VAL_90]], align 16
! LLVMIR:         %[[VAL_99:.*]] = bitcast <2 x i64> %[[VAL_95]] to <16 x i8>
! LLVMIR:         %[[VAL_100:.*]] = bitcast <2 x i64> %[[VAL_96]] to <16 x i8>
! LLVMIR:         %[[VAL_101:.*]] = bitcast <2 x i64> %[[VAL_97]] to <16 x i8>
! LLVMIR:         %[[VAL_102:.*]] = bitcast <2 x i64> %[[VAL_98]] to <16 x i8>
! LLVMIR:         %[[VAL_103:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_99]], <16 x i8> %[[VAL_100]], <16 x i8> %[[VAL_101]], <16 x i8> %[[VAL_102]])
! LLVMIR:         store <512 x i1> %[[VAL_103]], ptr %[[VAL_94]], align 64

      subroutine test_assemble_acc_r4()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_r4

! CHECK-LABEL: @test_assemble_acc_r4
! LLVMIR:         %[[VAL_104:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_105:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_106:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_107:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_108:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_109:.*]] = load <4 x float>, ptr %[[VAL_107]], align 16
! LLVMIR:         %[[VAL_110:.*]] = load <4 x float>, ptr %[[VAL_106]], align 16
! LLVMIR:         %[[VAL_111:.*]] = load <4 x float>, ptr %[[VAL_105]], align 16
! LLVMIR:         %[[VAL_112:.*]] = load <4 x float>, ptr %[[VAL_104]], align 16
! LLVMIR:         %[[VAL_113:.*]] = bitcast <4 x float> %[[VAL_109]] to <16 x i8>
! LLVMIR:         %[[VAL_114:.*]] = bitcast <4 x float> %[[VAL_110]] to <16 x i8>
! LLVMIR:         %[[VAL_115:.*]] = bitcast <4 x float> %[[VAL_111]] to <16 x i8>
! LLVMIR:         %[[VAL_116:.*]] = bitcast <4 x float> %[[VAL_112]] to <16 x i8>
! LLVMIR:         %[[VAL_117:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_113]], <16 x i8> %[[VAL_114]], <16 x i8> %[[VAL_115]], <16 x i8> %[[VAL_116]])
! LLVMIR:         store <512 x i1> %[[VAL_117]], ptr %[[VAL_108]], align 64

      subroutine test_assemble_acc_r8()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_assemble_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_assemble_acc_r8

!CHECK-LABEL: @test_assemble_acc_r8
! LLVMIR:         %[[VAL_118:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_119:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_120:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_121:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_122:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_123:.*]] = load <2 x double>, ptr %[[VAL_121]], align 16
! LLVMIR:         %[[VAL_124:.*]] = load <2 x double>, ptr %[[VAL_120]], align 16
! LLVMIR:         %[[VAL_125:.*]] = load <2 x double>, ptr %[[VAL_119]], align 16
! LLVMIR:         %[[VAL_126:.*]] = load <2 x double>, ptr %[[VAL_118]], align 16
! LLVMIR:         %[[VAL_127:.*]] = bitcast <2 x double> %[[VAL_123]] to <16 x i8>
! LLVMIR:         %[[VAL_128:.*]] = bitcast <2 x double> %[[VAL_124]] to <16 x i8>
! LLVMIR:         %[[VAL_129:.*]] = bitcast <2 x double> %[[VAL_125]] to <16 x i8>
! LLVMIR:         %[[VAL_130:.*]] = bitcast <2 x double> %[[VAL_126]] to <16 x i8>
! LLVMIR:         %[[VAL_131:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_127]], <16 x i8> %[[VAL_128]], <16 x i8> %[[VAL_129]], <16 x i8> %[[VAL_130]])
! LLVMIR:         store <512 x i1> %[[VAL_131]], ptr %[[VAL_122]], align 64

! mma_assemble_pair

      subroutine test_mma_assemble_pair_i1()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_i1

!LLVMIR: @test_mma_assemble_pair_i1_
! LLVMIR:         %[[VAL_132:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_133:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_134:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_135:.*]] = load <16 x i8>, ptr %[[VAL_134]], align 16
! LLVMIR:         %[[VAL_136:.*]] = load <16 x i8>, ptr %[[VAL_133]], align 16
! LLVMIR:         %[[VAL_137:.*]] = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %[[VAL_135]], <16 x i8> %[[VAL_136]])
! LLVMIR:         store <256 x i1> %[[VAL_137]], ptr %[[VAL_132]], align 32

      subroutine test_mma_assemble_pair_i2()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_i2

!LLVMIR: @test_mma_assemble_pair_i2_
! LLVMIR:         %[[VAL_138:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_139:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_140:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_141:.*]] = load <8 x i16>, ptr %[[VAL_140]], align 16
! LLVMIR:         %[[VAL_142:.*]] = load <8 x i16>, ptr %[[VAL_139]], align 16
! LLVMIR:         %[[VAL_143:.*]] = bitcast <8 x i16> %[[VAL_141]] to <16 x i8>
! LLVMIR:         %[[VAL_144:.*]] = bitcast <8 x i16> %[[VAL_142]] to <16 x i8>
! LLVMIR:         %[[VAL_145:.*]] = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %[[VAL_143]], <16 x i8> %[[VAL_144]])
! LLVMIR:         store <256 x i1> %[[VAL_145]], ptr %[[VAL_138]], align 32

      subroutine test_mma_assemble_pair_i4()
      use, intrinsic :: mma
      implicit none
      vector(integer(4)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_i4

!LLVMIR: @test_mma_assemble_pair_i4_
! LLVMIR:         %[[VAL_146:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_147:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_148:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_149:.*]] = load <4 x i32>, ptr %[[VAL_148]], align 16
! LLVMIR:         %[[VAL_150:.*]] = load <4 x i32>, ptr %[[VAL_147]], align 16
! LLVMIR:         %[[VAL_151:.*]] = bitcast <4 x i32> %[[VAL_149]] to <16 x i8>
! LLVMIR:         %[[VAL_152:.*]] = bitcast <4 x i32> %[[VAL_150]] to <16 x i8>
! LLVMIR:         %[[VAL_153:.*]] = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %[[VAL_151]], <16 x i8> %[[VAL_152]])
! LLVMIR:         store <256 x i1> %[[VAL_153]], ptr %[[VAL_146]], align 32

      subroutine test_mma_assemble_pair_i8()
      use, intrinsic :: mma
      implicit none
      vector(integer(8)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_i8

!LLVMIR: @test_mma_assemble_pair_i8_
! LLVMIR:         %[[VAL_154:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_155:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_156:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_157:.*]] = load <2 x i64>, ptr %[[VAL_156]], align 16
! LLVMIR:         %[[VAL_158:.*]] = load <2 x i64>, ptr %[[VAL_155]], align 16
! LLVMIR:         %[[VAL_159:.*]] = bitcast <2 x i64> %[[VAL_157]] to <16 x i8>
! LLVMIR:         %[[VAL_160:.*]] = bitcast <2 x i64> %[[VAL_158]] to <16 x i8>
! LLVMIR:         %[[VAL_161:.*]] = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %[[VAL_159]], <16 x i8> %[[VAL_160]])
! LLVMIR:         store <256 x i1> %[[VAL_161]], ptr %[[VAL_154]], align 32

      subroutine test_mma_assemble_pair_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_u1

!LLVMIR: @test_mma_assemble_pair_u1_
! LLVMIR:         %[[VAL_162:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_163:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_164:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_165:.*]] = load <16 x i8>, ptr %[[VAL_164]], align 16
! LLVMIR:         %[[VAL_166:.*]] = load <16 x i8>, ptr %[[VAL_163]], align 16
! LLVMIR:         %[[VAL_167:.*]] = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %[[VAL_165]], <16 x i8> %[[VAL_166]])
! LLVMIR:         store <256 x i1> %[[VAL_167]], ptr %[[VAL_162]], align 32

      subroutine test_mma_assemble_pair_u2()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(2)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_u2

!LLVMIR: @test_mma_assemble_pair_u2_
! LLVMIR:         %[[VAL_168:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_169:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_170:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_171:.*]] = load <8 x i16>, ptr %[[VAL_170]], align 16
! LLVMIR:         %[[VAL_172:.*]] = load <8 x i16>, ptr %[[VAL_169]], align 16
! LLVMIR:         %[[VAL_173:.*]] = bitcast <8 x i16> %[[VAL_171]] to <16 x i8>
! LLVMIR:         %[[VAL_174:.*]] = bitcast <8 x i16> %[[VAL_172]] to <16 x i8>
! LLVMIR:         %[[VAL_175:.*]] = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %[[VAL_173]], <16 x i8> %[[VAL_174]])
! LLVMIR:         store <256 x i1> %[[VAL_175]], ptr %[[VAL_168]], align 32

      subroutine test_mma_assemble_pair_u4()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(4)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_u4

!LLVMIR: @test_mma_assemble_pair_u4_
! LLVMIR:         %[[VAL_176:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_177:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_178:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_179:.*]] = load <4 x i32>, ptr %[[VAL_178]], align 16
! LLVMIR:         %[[VAL_180:.*]] = load <4 x i32>, ptr %[[VAL_177]], align 16
! LLVMIR:         %[[VAL_181:.*]] = bitcast <4 x i32> %[[VAL_179]] to <16 x i8>
! LLVMIR:         %[[VAL_182:.*]] = bitcast <4 x i32> %[[VAL_180]] to <16 x i8>
! LLVMIR:         %[[VAL_183:.*]] = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %[[VAL_181]], <16 x i8> %[[VAL_182]])
! LLVMIR:         store <256 x i1> %[[VAL_183]], ptr %[[VAL_176]], align 32

      subroutine test_mma_assemble_pair_u8()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(8)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_u8

!LLVMIR: @test_mma_assemble_pair_u8_
! LLVMIR:         %[[VAL_184:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_185:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_186:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_187:.*]] = load <2 x i64>, ptr %[[VAL_186]], align 16
! LLVMIR:         %[[VAL_188:.*]] = load <2 x i64>, ptr %[[VAL_185]], align 16
! LLVMIR:         %[[VAL_189:.*]] = bitcast <2 x i64> %[[VAL_187]] to <16 x i8>
! LLVMIR:         %[[VAL_190:.*]] = bitcast <2 x i64> %[[VAL_188]] to <16 x i8>
! LLVMIR:         %[[VAL_191:.*]] = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %[[VAL_189]], <16 x i8> %[[VAL_190]])
! LLVMIR:         store <256 x i1> %[[VAL_191]], ptr %[[VAL_184]], align 32

      subroutine test_mma_assemble_pair_r4()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_r4

!LLVMIR: @test_mma_assemble_pair_r4_
! LLVMIR:         %[[VAL_192:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_193:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_194:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_195:.*]] = load <4 x float>, ptr %[[VAL_194]], align 16
! LLVMIR:         %[[VAL_196:.*]] = load <4 x float>, ptr %[[VAL_193]], align 16
! LLVMIR:         %[[VAL_197:.*]] = bitcast <4 x float> %[[VAL_195]] to <16 x i8>
! LLVMIR:         %[[VAL_198:.*]] = bitcast <4 x float> %[[VAL_196]] to <16 x i8>
! LLVMIR:         %[[VAL_199:.*]] = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %[[VAL_197]], <16 x i8> %[[VAL_198]])
! LLVMIR:         store <256 x i1> %[[VAL_199]], ptr %[[VAL_192]], align 32

      subroutine test_mma_assemble_pair_r8()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vi10, vi11
      __vector_pair :: vp
      call mma_assemble_pair(vp, vi10, vi11)
      end subroutine test_mma_assemble_pair_r8

!LLVMIR: @test_mma_assemble_pair_r8_
! LLVMIR:         %[[VAL_200:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_201:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_202:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_203:.*]] = load <2 x double>, ptr %[[VAL_202]], align 16
! LLVMIR:         %[[VAL_204:.*]] = load <2 x double>, ptr %[[VAL_201]], align 16
! LLVMIR:         %[[VAL_205:.*]] = bitcast <2 x double> %[[VAL_203]] to <16 x i8>
! LLVMIR:         %[[VAL_206:.*]] = bitcast <2 x double> %[[VAL_204]] to <16 x i8>
! LLVMIR:         %[[VAL_207:.*]] = call <256 x i1> @llvm.ppc.vsx.assemble.pair(<16 x i8> %[[VAL_205]], <16 x i8> %[[VAL_206]])
! LLVMIR:         store <256 x i1> %[[VAL_207]], ptr %[[VAL_200]], align 32

! mma_disassemble_acc

      subroutine test_mma_build_acc_i1()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_i1

!CHECK-LABEL: @test_mma_build_acc_i1
! LLVMIR:         %[[VAL_208:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_209:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_210:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_211:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_212:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_213:.*]] = load <16 x i8>, ptr %[[VAL_211]], align 16
! LLVMIR:         %[[VAL_214:.*]] = load <16 x i8>, ptr %[[VAL_210]], align 16
! LLVMIR:         %[[VAL_215:.*]] = load <16 x i8>, ptr %[[VAL_209]], align 16
! LLVMIR:         %[[VAL_216:.*]] = load <16 x i8>, ptr %[[VAL_208]], align 16
! LLVMIR:         %[[VAL_217:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_216]], <16 x i8> %[[VAL_215]], <16 x i8> %[[VAL_214]], <16 x i8> %[[VAL_213]])
! LLVMIR:         store <512 x i1> %[[VAL_217]], ptr %[[VAL_212]], align 64

      subroutine test_mma_build_acc_i2()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_i2

!CHECK-LABEL: @test_mma_build_acc_i2
! LLVMIR:         %[[VAL_218:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_219:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_220:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_221:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_222:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_223:.*]] = load <8 x i16>, ptr %[[VAL_221]], align 16
! LLVMIR:         %[[VAL_224:.*]] = load <8 x i16>, ptr %[[VAL_220]], align 16
! LLVMIR:         %[[VAL_225:.*]] = load <8 x i16>, ptr %[[VAL_219]], align 16
! LLVMIR:         %[[VAL_226:.*]] = load <8 x i16>, ptr %[[VAL_218]], align 16
! LLVMIR:         %[[VAL_227:.*]] = bitcast <8 x i16> %[[VAL_226]] to <16 x i8>
! LLVMIR:         %[[VAL_228:.*]] = bitcast <8 x i16> %[[VAL_225]] to <16 x i8>
! LLVMIR:         %[[VAL_229:.*]] = bitcast <8 x i16> %[[VAL_224]] to <16 x i8>
! LLVMIR:         %[[VAL_230:.*]] = bitcast <8 x i16> %[[VAL_223]] to <16 x i8>
! LLVMIR:         %[[VAL_231:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_227]], <16 x i8> %[[VAL_228]], <16 x i8> %[[VAL_229]], <16 x i8> %[[VAL_230]])
! LLVMIR:         store <512 x i1> %[[VAL_231]], ptr %[[VAL_222]], align 64

      subroutine test_mma_build_acc_i4()
      use, intrinsic :: mma
      implicit none
      vector(integer(4)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_i4

!CHECK-LABEL: @test_mma_build_acc_i4
! LLVMIR:         %[[VAL_232:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_233:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_234:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_235:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_236:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_237:.*]] = load <4 x i32>, ptr %[[VAL_235]], align 16
! LLVMIR:         %[[VAL_238:.*]] = load <4 x i32>, ptr %[[VAL_234]], align 16
! LLVMIR:         %[[VAL_239:.*]] = load <4 x i32>, ptr %[[VAL_233]], align 16
! LLVMIR:         %[[VAL_240:.*]] = load <4 x i32>, ptr %[[VAL_232]], align 16
! LLVMIR:         %[[VAL_241:.*]] = bitcast <4 x i32> %[[VAL_240]] to <16 x i8>
! LLVMIR:         %[[VAL_242:.*]] = bitcast <4 x i32> %[[VAL_239]] to <16 x i8>
! LLVMIR:         %[[VAL_243:.*]] = bitcast <4 x i32> %[[VAL_238]] to <16 x i8>
! LLVMIR:         %[[VAL_244:.*]] = bitcast <4 x i32> %[[VAL_237]] to <16 x i8>
! LLVMIR:         %[[VAL_245:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_241]], <16 x i8> %[[VAL_242]], <16 x i8> %[[VAL_243]], <16 x i8> %[[VAL_244]])
! LLVMIR:         store <512 x i1> %[[VAL_245]], ptr %[[VAL_236]], align 64

      subroutine test_mma_build_acc_i8()
      use, intrinsic :: mma
      implicit none
      vector(integer(8)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_i8

!CHECK-LABEL: @test_mma_build_acc_i8
! LLVMIR:         %[[VAL_246:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_247:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_248:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_249:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_250:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_251:.*]] = load <2 x i64>, ptr %[[VAL_249]], align 16
! LLVMIR:         %[[VAL_252:.*]] = load <2 x i64>, ptr %[[VAL_248]], align 16
! LLVMIR:         %[[VAL_253:.*]] = load <2 x i64>, ptr %[[VAL_247]], align 16
! LLVMIR:         %[[VAL_254:.*]] = load <2 x i64>, ptr %[[VAL_246]], align 16
! LLVMIR:         %[[VAL_255:.*]] = bitcast <2 x i64> %[[VAL_254]] to <16 x i8>
! LLVMIR:         %[[VAL_256:.*]] = bitcast <2 x i64> %[[VAL_253]] to <16 x i8>
! LLVMIR:         %[[VAL_257:.*]] = bitcast <2 x i64> %[[VAL_252]] to <16 x i8>
! LLVMIR:         %[[VAL_258:.*]] = bitcast <2 x i64> %[[VAL_251]] to <16 x i8>
! LLVMIR:         %[[VAL_259:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_255]], <16 x i8> %[[VAL_256]], <16 x i8> %[[VAL_257]], <16 x i8> %[[VAL_258]])
! LLVMIR:         store <512 x i1> %[[VAL_259]], ptr %[[VAL_250]], align 64

      subroutine test_mma_build_acc_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_u1

!CHECK-LABEL: @test_mma_build_acc_u1
! LLVMIR:         %[[VAL_260:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_261:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_262:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_263:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_264:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_265:.*]] = load <16 x i8>, ptr %[[VAL_263]], align 16
! LLVMIR:         %[[VAL_266:.*]] = load <16 x i8>, ptr %[[VAL_262]], align 16
! LLVMIR:         %[[VAL_267:.*]] = load <16 x i8>, ptr %[[VAL_261]], align 16
! LLVMIR:         %[[VAL_268:.*]] = load <16 x i8>, ptr %[[VAL_260]], align 16
! LLVMIR:         %[[VAL_269:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_268]], <16 x i8> %[[VAL_267]], <16 x i8> %[[VAL_266]], <16 x i8> %[[VAL_265]])
! LLVMIR:         store <512 x i1> %[[VAL_269]], ptr %[[VAL_264]], align 64

      subroutine test_mma_build_acc_u2()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(2)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_u2

!CHECK-LABEL: @test_mma_build_acc_u2
! LLVMIR:         %[[VAL_270:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_271:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_272:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_273:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_274:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_275:.*]] = load <8 x i16>, ptr %[[VAL_273]], align 16
! LLVMIR:         %[[VAL_276:.*]] = load <8 x i16>, ptr %[[VAL_272]], align 16
! LLVMIR:         %[[VAL_277:.*]] = load <8 x i16>, ptr %[[VAL_271]], align 16
! LLVMIR:         %[[VAL_278:.*]] = load <8 x i16>, ptr %[[VAL_270]], align 16
! LLVMIR:         %[[VAL_279:.*]] = bitcast <8 x i16> %[[VAL_278]] to <16 x i8>
! LLVMIR:         %[[VAL_280:.*]] = bitcast <8 x i16> %[[VAL_277]] to <16 x i8>
! LLVMIR:         %[[VAL_281:.*]] = bitcast <8 x i16> %[[VAL_276]] to <16 x i8>
! LLVMIR:         %[[VAL_282:.*]] = bitcast <8 x i16> %[[VAL_275]] to <16 x i8>
! LLVMIR:         %[[VAL_283:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_279]], <16 x i8> %[[VAL_280]], <16 x i8> %[[VAL_281]], <16 x i8> %[[VAL_282]])
! LLVMIR:         store <512 x i1> %[[VAL_283]], ptr %[[VAL_274]], align 64

      subroutine test_mma_build_acc_u4()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(4)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_u4

!CHECK-LABEL: @test_mma_build_acc_u4
! LLVMIR:         %[[VAL_284:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_285:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_286:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_287:.*]] = alloca <4 x i32>, i64 1, align 16
! LLVMIR:         %[[VAL_288:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_289:.*]] = load <4 x i32>, ptr %[[VAL_287]], align 16
! LLVMIR:         %[[VAL_290:.*]] = load <4 x i32>, ptr %[[VAL_286]], align 16
! LLVMIR:         %[[VAL_291:.*]] = load <4 x i32>, ptr %[[VAL_285]], align 16
! LLVMIR:         %[[VAL_292:.*]] = load <4 x i32>, ptr %[[VAL_284]], align 16
! LLVMIR:         %[[VAL_293:.*]] = bitcast <4 x i32> %[[VAL_292]] to <16 x i8>
! LLVMIR:         %[[VAL_294:.*]] = bitcast <4 x i32> %[[VAL_291]] to <16 x i8>
! LLVMIR:         %[[VAL_295:.*]] = bitcast <4 x i32> %[[VAL_290]] to <16 x i8>
! LLVMIR:         %[[VAL_296:.*]] = bitcast <4 x i32> %[[VAL_289]] to <16 x i8>
! LLVMIR:         %[[VAL_297:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_293]], <16 x i8> %[[VAL_294]], <16 x i8> %[[VAL_295]], <16 x i8> %[[VAL_296]])
! LLVMIR:         store <512 x i1> %[[VAL_297]], ptr %[[VAL_288]], align 64

      subroutine test_mma_build_acc_u8()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(8)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_u8

!CHECK-LABEL: @test_mma_build_acc_u8
! LLVMIR:         %[[VAL_298:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_299:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_300:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_301:.*]] = alloca <2 x i64>, i64 1, align 16
! LLVMIR:         %[[VAL_302:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_303:.*]] = load <2 x i64>, ptr %[[VAL_301]], align 16
! LLVMIR:         %[[VAL_304:.*]] = load <2 x i64>, ptr %[[VAL_300]], align 16
! LLVMIR:         %[[VAL_305:.*]] = load <2 x i64>, ptr %[[VAL_299]], align 16
! LLVMIR:         %[[VAL_306:.*]] = load <2 x i64>, ptr %[[VAL_298]], align 16
! LLVMIR:         %[[VAL_307:.*]] = bitcast <2 x i64> %[[VAL_306]] to <16 x i8>
! LLVMIR:         %[[VAL_308:.*]] = bitcast <2 x i64> %[[VAL_305]] to <16 x i8>
! LLVMIR:         %[[VAL_309:.*]] = bitcast <2 x i64> %[[VAL_304]] to <16 x i8>
! LLVMIR:         %[[VAL_310:.*]] = bitcast <2 x i64> %[[VAL_303]] to <16 x i8>
! LLVMIR:         %[[VAL_311:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_307]], <16 x i8> %[[VAL_308]], <16 x i8> %[[VAL_309]], <16 x i8> %[[VAL_310]])
! LLVMIR:         store <512 x i1> %[[VAL_311]], ptr %[[VAL_302]], align 64


      subroutine test_mma_build_acc_r4()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_r4

!CHECK-LABEL: @test_mma_build_acc_r4
! LLVMIR:         %[[VAL_312:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_313:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_314:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_315:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_316:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_317:.*]] = load <4 x float>, ptr %[[VAL_315]], align 16
! LLVMIR:         %[[VAL_318:.*]] = load <4 x float>, ptr %[[VAL_314]], align 16
! LLVMIR:         %[[VAL_319:.*]] = load <4 x float>, ptr %[[VAL_313]], align 16
! LLVMIR:         %[[VAL_320:.*]] = load <4 x float>, ptr %[[VAL_312]], align 16
! LLVMIR:         %[[VAL_321:.*]] = bitcast <4 x float> %[[VAL_320]] to <16 x i8>
! LLVMIR:         %[[VAL_322:.*]] = bitcast <4 x float> %[[VAL_319]] to <16 x i8>
! LLVMIR:         %[[VAL_323:.*]] = bitcast <4 x float> %[[VAL_318]] to <16 x i8>
! LLVMIR:         %[[VAL_324:.*]] = bitcast <4 x float> %[[VAL_317]] to <16 x i8>
! LLVMIR:         %[[VAL_325:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_321]], <16 x i8> %[[VAL_322]], <16 x i8> %[[VAL_323]], <16 x i8> %[[VAL_324]])
! LLVMIR:         store <512 x i1> %[[VAL_325]], ptr %[[VAL_316]], align 64


      subroutine test_mma_build_acc_r8()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vi10, vi11, vi12, vi13
      __vector_quad :: cq
      call mma_build_acc(cq, vi10, vi11, vi12, vi13)
      end subroutine test_mma_build_acc_r8

!CHECK-LABEL: @test_mma_build_acc_r8
! LLVMIR:         %[[VAL_326:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_327:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_328:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_329:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_330:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_331:.*]] = load <2 x double>, ptr %[[VAL_329]], align 16
! LLVMIR:         %[[VAL_332:.*]] = load <2 x double>, ptr %[[VAL_328]], align 16
! LLVMIR:         %[[VAL_333:.*]] = load <2 x double>, ptr %[[VAL_327]], align 16
! LLVMIR:         %[[VAL_334:.*]] = load <2 x double>, ptr %[[VAL_326]], align 16
! LLVMIR:         %[[VAL_335:.*]] = bitcast <2 x double> %[[VAL_334]] to <16 x i8>
! LLVMIR:         %[[VAL_336:.*]] = bitcast <2 x double> %[[VAL_333]] to <16 x i8>
! LLVMIR:         %[[VAL_337:.*]] = bitcast <2 x double> %[[VAL_332]] to <16 x i8>
! LLVMIR:         %[[VAL_338:.*]] = bitcast <2 x double> %[[VAL_331]] to <16 x i8>
! LLVMIR:         %[[VAL_339:.*]] = call <512 x i1> @llvm.ppc.mma.assemble.acc(<16 x i8> %[[VAL_335]], <16 x i8> %[[VAL_336]], <16 x i8> %[[VAL_337]], <16 x i8> %[[VAL_338]])
! LLVMIR:         store <512 x i1> %[[VAL_339]], ptr %[[VAL_330]], align 64

! mma_disassemble_acc

      subroutine test_disassemble_acc()
      use, intrinsic :: mma
      implicit none
      __vector_quad :: vq
      real :: data
      call mma_disassemble_acc(data, vq)
      end subroutine

!CHECK-LABEL: @test_disassemble_acc_
! LLVMIR:         %[[VAL_340:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_341:.*]] = alloca float, i64 1, align 4
! LLVMIR:         %[[VAL_342:.*]] = load <512 x i1>, ptr %[[VAL_340]], align 64
! LLVMIR:         %[[VAL_343:.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.ppc.mma.disassemble.acc(<512 x i1> %[[VAL_342]])
! LLVMIR:         store { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %[[VAL_343]], ptr %[[VAL_341]], align 16

! mma_disassemble_pair

      subroutine test_disassemble_pair()
      use, intrinsic :: mma
      implicit none
      __vector_pair :: vp
      real :: data
      call mma_disassemble_pair(data, vp)
      end subroutine

!CHECK-LABEL: @test_disassemble_pair_
! LLVMIR:         %[[VAL_344:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_345:.*]] = alloca float, i64 1, align 4
! LLVMIR:         %[[VAL_346:.*]] = load <256 x i1>, ptr %[[VAL_344]], align 32
! LLVMIR:         %[[VAL_347:.*]] = call { <16 x i8>, <16 x i8> } @llvm.ppc.vsx.disassemble.pair(<256 x i1> %[[VAL_346]])
! LLVMIR:         store { <16 x i8>, <16 x i8> } %[[VAL_347]], ptr %[[VAL_345]], align 16
