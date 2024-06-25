! RUN: %flang_fc1 -flang-experimental-hlfir -triple powerpc64le-unknown-unknown -target-cpu pwr10 -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

      subroutine test_xvbf16ger2()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvbf16ger2(cq, vu10, vu11)
      end subroutine test_xvbf16ger2

!CHECK-LABEL: @test_xvbf16ger2_
! LLVMIR:         %[[VAL_0:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_1:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_2:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_3:.*]] = load <16 x i8>, ptr %[[VAL_1]], align 16
! LLVMIR:         %[[VAL_4:.*]] = load <16 x i8>, ptr %[[VAL_0]], align 16
! LLVMIR:         %[[VAL_5:.*]] = call <512 x i1> @llvm.ppc.mma.xvbf16ger2(<16 x i8> %[[VAL_3]], <16 x i8> %[[VAL_4]])
! LLVMIR:         store <512 x i1> %[[VAL_5]], ptr %[[VAL_2]], align 64


      subroutine test_xvbf16ger2nn()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvbf16ger2nn(cq, vu10, vu11)
      end subroutine test_xvbf16ger2nn

!CHECK-LABEL: @test_xvbf16ger2nn_
! LLVMIR:         %[[VAL_6:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_7:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_8:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_9:.*]] = load <16 x i8>, ptr %[[VAL_7]], align 16
! LLVMIR:         %[[VAL_10:.*]] = load <16 x i8>, ptr %[[VAL_6]], align 16
! LLVMIR:         %[[VAL_11:.*]] = load <512 x i1>, ptr %[[VAL_8]], align 64
! LLVMIR:         %[[VAL_12:.*]] = call <512 x i1> @llvm.ppc.mma.xvbf16ger2nn(<512 x i1> %[[VAL_11]], <16 x i8> %[[VAL_9]], <16 x i8> %[[VAL_10]])
! LLVMIR:         store <512 x i1> %[[VAL_12]], ptr %[[VAL_8]], align 64

      subroutine test_xvbf16ger2np()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvbf16ger2np(cq, vu10, vu11)
      end subroutine test_xvbf16ger2np

!CHECK-LABEL: @test_xvbf16ger2np_
! LLVMIR:         %[[VAL_13:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_14:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_15:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_16:.*]] = load <16 x i8>, ptr %[[VAL_14]], align 16
! LLVMIR:         %[[VAL_17:.*]] = load <16 x i8>, ptr %[[VAL_13]], align 16
! LLVMIR:         %[[VAL_18:.*]] = load <512 x i1>, ptr %[[VAL_15]], align 64
! LLVMIR:         %[[VAL_19:.*]] = call <512 x i1> @llvm.ppc.mma.xvbf16ger2np(<512 x i1> %[[VAL_18]], <16 x i8> %[[VAL_16]], <16 x i8> %[[VAL_17]])
! LLVMIR:         store <512 x i1> %[[VAL_19]], ptr %[[VAL_15]], align 64

      subroutine test_xvbf16ger2pn()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvbf16ger2pn(cq, vu10, vu11)
      end subroutine test_xvbf16ger2pn

!CHECK-LABEL: @test_xvbf16ger2pn_
! LLVMIR:         %[[VAL_20:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_21:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_22:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_23:.*]] = load <16 x i8>, ptr %[[VAL_21]], align 16
! LLVMIR:         %[[VAL_24:.*]] = load <16 x i8>, ptr %[[VAL_20]], align 16
! LLVMIR:         %[[VAL_25:.*]] = load <512 x i1>, ptr %[[VAL_22]], align 64
! LLVMIR:         %[[VAL_26:.*]] = call <512 x i1> @llvm.ppc.mma.xvbf16ger2pn(<512 x i1> %[[VAL_25]], <16 x i8> %[[VAL_23]], <16 x i8> %[[VAL_24]])
! LLVMIR:         store <512 x i1> %[[VAL_26]], ptr %[[VAL_22]], align 64

      subroutine test_xvbf16ger2pp()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvbf16ger2pp(cq, vu10, vu11)
      end subroutine test_xvbf16ger2pp

!CHECK-LABEL: @test_xvbf16ger2pp_
! LLVMIR:         %[[VAL_27:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_28:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_29:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_30:.*]] = load <16 x i8>, ptr %[[VAL_28]], align 16
! LLVMIR:         %[[VAL_31:.*]] = load <16 x i8>, ptr %[[VAL_27]], align 16
! LLVMIR:         %[[VAL_32:.*]] = load <512 x i1>, ptr %[[VAL_29]], align 64
! LLVMIR:         %[[VAL_33:.*]] = call <512 x i1> @llvm.ppc.mma.xvbf16ger2pp(<512 x i1> %[[VAL_32]], <16 x i8> %[[VAL_30]], <16 x i8> %[[VAL_31]])
! LLVMIR:         store <512 x i1> %[[VAL_33]], ptr %[[VAL_29]], align 64

      subroutine test_xvf16ger2()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf16ger2(cq, vu10, vu11)
      end subroutine test_xvf16ger2

!CHECK-LABEL: @test_xvf16ger2_
! LLVMIR:         %[[VAL_34:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_35:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_36:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_37:.*]] = load <16 x i8>, ptr %[[VAL_35]], align 16
! LLVMIR:         %[[VAL_38:.*]] = load <16 x i8>, ptr %[[VAL_34]], align 16
! LLVMIR:         %[[VAL_39:.*]] = call <512 x i1> @llvm.ppc.mma.xvf16ger2(<16 x i8> %[[VAL_37]], <16 x i8> %[[VAL_38]])
! LLVMIR:         store <512 x i1> %[[VAL_39]], ptr %[[VAL_36]], align 64

      subroutine test_xvf16ger2nn()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf16ger2nn(cq, vu10, vu11)
      end subroutine test_xvf16ger2nn

!CHECK-LABEL: @test_xvf16ger2nn_
! LLVMIR:         %[[VAL_40:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_41:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_42:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_43:.*]] = load <16 x i8>, ptr %[[VAL_41]], align 16
! LLVMIR:         %[[VAL_44:.*]] = load <16 x i8>, ptr %[[VAL_40]], align 16
! LLVMIR:         %[[VAL_45:.*]] = load <512 x i1>, ptr %[[VAL_42]], align 64
! LLVMIR:         %[[VAL_46:.*]] = call <512 x i1> @llvm.ppc.mma.xvf16ger2nn(<512 x i1> %[[VAL_45]], <16 x i8> %[[VAL_43]], <16 x i8> %[[VAL_44]])
! LLVMIR:         store <512 x i1> %[[VAL_46]], ptr %[[VAL_42]], align 64

      subroutine test_xvf16ger2np()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf16ger2np(cq, vu10, vu11)
      end subroutine test_xvf16ger2np

!CHECK-LABEL: @test_xvf16ger2np_
! LLVMIR:         %[[VAL_47:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_48:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_49:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_50:.*]] = load <16 x i8>, ptr %[[VAL_48]], align 16
! LLVMIR:         %[[VAL_51:.*]] = load <16 x i8>, ptr %[[VAL_47]], align 16
! LLVMIR:         %[[VAL_52:.*]] = load <512 x i1>, ptr %[[VAL_49]], align 64
! LLVMIR:         %[[VAL_53:.*]] = call <512 x i1> @llvm.ppc.mma.xvf16ger2np(<512 x i1> %[[VAL_52]], <16 x i8> %[[VAL_50]], <16 x i8> %[[VAL_51]])
! LLVMIR:         store <512 x i1> %[[VAL_53]], ptr %[[VAL_49]], align 64

      subroutine test_xvf16ger2pn()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf16ger2pn(cq, vu10, vu11)
      end subroutine test_xvf16ger2pn

!CHECK-LABEL: @test_xvf16ger2pn_
! LLVMIR:         %[[VAL_54:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_55:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_56:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_57:.*]] = load <16 x i8>, ptr %[[VAL_55]], align 16
! LLVMIR:         %[[VAL_58:.*]] = load <16 x i8>, ptr %[[VAL_54]], align 16
! LLVMIR:         %[[VAL_59:.*]] = load <512 x i1>, ptr %[[VAL_56]], align 64
! LLVMIR:         %[[VAL_60:.*]] = call <512 x i1> @llvm.ppc.mma.xvf16ger2pn(<512 x i1> %[[VAL_59]], <16 x i8> %[[VAL_57]], <16 x i8> %[[VAL_58]])
! LLVMIR:         store <512 x i1> %[[VAL_60]], ptr %[[VAL_56]], align 64

      subroutine test_xvf16ger2pp()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf16ger2pp(cq, vu10, vu11)
      end subroutine test_xvf16ger2pp

!CHECK-LABEL: @test_xvf16ger2pp_
! LLVMIR:         %[[VAL_61:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_62:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_63:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_64:.*]] = load <16 x i8>, ptr %[[VAL_62]], align 16
! LLVMIR:         %[[VAL_65:.*]] = load <16 x i8>, ptr %[[VAL_61]], align 16
! LLVMIR:         %[[VAL_66:.*]] = load <512 x i1>, ptr %[[VAL_63]], align 64
! LLVMIR:         %[[VAL_67:.*]] = call <512 x i1> @llvm.ppc.mma.xvf16ger2pp(<512 x i1> %[[VAL_66]], <16 x i8> %[[VAL_64]], <16 x i8> %[[VAL_65]])
! LLVMIR:         store <512 x i1> %[[VAL_67]], ptr %[[VAL_63]], align 64

      subroutine test_xvf32ger_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf32ger(cq, vu10, vu11)
      end subroutine test_xvf32ger_u1

!CHECK-LABEL: @test_xvf32ger_u1_
! LLVMIR:         %[[VAL_68:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_69:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_70:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_71:.*]] = load <16 x i8>, ptr %[[VAL_69]], align 16
! LLVMIR:         %[[VAL_72:.*]] = load <16 x i8>, ptr %[[VAL_68]], align 16
! LLVMIR:         %[[VAL_73:.*]] = call <512 x i1> @llvm.ppc.mma.xvf32ger(<16 x i8> %[[VAL_71]], <16 x i8> %[[VAL_72]])
! LLVMIR:         store <512 x i1> %[[VAL_73]], ptr %[[VAL_70]], align 64


      subroutine test_xvf32ger_r4()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_xvf32ger(cq, vr40, vr41)
      end subroutine test_xvf32ger_r4

!CHECK-LABEL: @test_xvf32ger_r4_
! LLVMIR:         %[[VAL_74:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_75:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_76:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_77:.*]] = load <4 x float>, ptr %[[VAL_75]], align 16
! LLVMIR:         %[[VAL_78:.*]] = load <4 x float>, ptr %[[VAL_74]], align 16
! LLVMIR:         %[[VAL_79:.*]] = bitcast <4 x float> %[[VAL_77]] to <16 x i8>
! LLVMIR:         %[[VAL_80:.*]] = bitcast <4 x float> %[[VAL_78]] to <16 x i8>
! LLVMIR:         %[[VAL_81:.*]] = call <512 x i1> @llvm.ppc.mma.xvf32ger(<16 x i8> %[[VAL_79]], <16 x i8> %[[VAL_80]])
! LLVMIR:         store <512 x i1> %[[VAL_81]], ptr %[[VAL_76]], align 64

      subroutine test_xvf32gernn_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf32gernn(cq, vu10, vu11)
      end subroutine test_xvf32gernn_u1

!CHECK-LABEL: @test_xvf32gernn_u1_
! LLVMIR:         %[[VAL_82:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_83:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_84:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_85:.*]] = load <16 x i8>, ptr %[[VAL_83]], align 16
! LLVMIR:         %[[VAL_86:.*]] = load <16 x i8>, ptr %[[VAL_82]], align 16
! LLVMIR:         %[[VAL_87:.*]] = load <512 x i1>, ptr %[[VAL_84]], align 64
! LLVMIR:         %[[VAL_88:.*]] = call <512 x i1> @llvm.ppc.mma.xvf32gernn(<512 x i1> %[[VAL_87]], <16 x i8> %[[VAL_85]], <16 x i8> %[[VAL_86]])
! LLVMIR:         store <512 x i1> %[[VAL_88]], ptr %[[VAL_84]], align 64

      subroutine test_xvf32gernn_r4()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_xvf32gernn(cq, vr40, vr41)
      end subroutine test_xvf32gernn_r4

!CHECK-LABEL: @test_xvf32gernn_r4_
! LLVMIR:         %[[VAL_89:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_90:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_91:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_92:.*]] = load <4 x float>, ptr %[[VAL_90]], align 16
! LLVMIR:         %[[VAL_93:.*]] = load <4 x float>, ptr %[[VAL_89]], align 16
! LLVMIR:         %[[VAL_94:.*]] = load <512 x i1>, ptr %[[VAL_91]], align 64
! LLVMIR:         %[[VAL_95:.*]] = bitcast <4 x float> %[[VAL_92]] to <16 x i8>
! LLVMIR:         %[[VAL_96:.*]] = bitcast <4 x float> %[[VAL_93]] to <16 x i8>
! LLVMIR:         %[[VAL_97:.*]] = call <512 x i1> @llvm.ppc.mma.xvf32gernn(<512 x i1> %[[VAL_94]], <16 x i8> %[[VAL_95]], <16 x i8> %[[VAL_96]])
! LLVMIR:         store <512 x i1> %[[VAL_97]], ptr %[[VAL_91]], align 64

      subroutine test_xvf32gernp_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf32gernp(cq, vu10, vu11)
      end subroutine test_xvf32gernp_u1

!CHECK-LABEL: @test_xvf32gernp_u1_
! LLVMIR:         %[[VAL_98:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_99:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_100:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_101:.*]] = load <16 x i8>, ptr %[[VAL_99]], align 16
! LLVMIR:         %[[VAL_102:.*]] = load <16 x i8>, ptr %[[VAL_98]], align 16
! LLVMIR:         %[[VAL_103:.*]] = load <512 x i1>, ptr %[[VAL_100]], align 64
! LLVMIR:         %[[VAL_104:.*]] = call <512 x i1> @llvm.ppc.mma.xvf32gernp(<512 x i1> %[[VAL_103]], <16 x i8> %[[VAL_101]], <16 x i8> %[[VAL_102]])
! LLVMIR:         store <512 x i1> %[[VAL_104]], ptr %[[VAL_100]], align 64

      subroutine test_xvf32gernp_r4()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_xvf32gernp(cq, vr40, vr41)
      end subroutine test_xvf32gernp_r4

!CHECK-LABEL: @test_xvf32gernp_r4_
! LLVMIR:         %[[VAL_105:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_106:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_107:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_108:.*]] = load <4 x float>, ptr %[[VAL_106]], align 16
! LLVMIR:         %[[VAL_109:.*]] = load <4 x float>, ptr %[[VAL_105]], align 16
! LLVMIR:         %[[VAL_110:.*]] = load <512 x i1>, ptr %[[VAL_107]], align 64
! LLVMIR:         %[[VAL_111:.*]] = bitcast <4 x float> %[[VAL_108]] to <16 x i8>
! LLVMIR:         %[[VAL_112:.*]] = bitcast <4 x float> %[[VAL_109]] to <16 x i8>
! LLVMIR:         %[[VAL_113:.*]] = call <512 x i1> @llvm.ppc.mma.xvf32gernp(<512 x i1> %[[VAL_110]], <16 x i8> %[[VAL_111]], <16 x i8> %[[VAL_112]])
! LLVMIR:         store <512 x i1> %[[VAL_113]], ptr %[[VAL_107]], align 64

      subroutine test_xvf32gerpn_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf32gerpn(cq, vu10, vu11)
      end subroutine test_xvf32gerpn_u1

!CHECK-LABEL: @test_xvf32gerpn_u1_
! LLVMIR:         %[[VAL_114:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_115:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_116:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_117:.*]] = load <16 x i8>, ptr %[[VAL_115]], align 16
! LLVMIR:         %[[VAL_118:.*]] = load <16 x i8>, ptr %[[VAL_114]], align 16
! LLVMIR:         %[[VAL_119:.*]] = load <512 x i1>, ptr %[[VAL_116]], align 64
! LLVMIR:         %[[VAL_120:.*]] = call <512 x i1> @llvm.ppc.mma.xvf32gerpn(<512 x i1> %[[VAL_119]], <16 x i8> %[[VAL_117]], <16 x i8> %[[VAL_118]])
! LLVMIR:         store <512 x i1> %[[VAL_120]], ptr %[[VAL_116]], align 64

      subroutine test_xvf32gerpn_r4()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_xvf32gerpn(cq, vr40, vr41)
      end subroutine test_xvf32gerpn_r4

!CHECK-LABEL: @test_xvf32gerpn_r4_
! LLVMIR:         %[[VAL_121:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_122:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_123:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_124:.*]] = load <4 x float>, ptr %[[VAL_122]], align 16
! LLVMIR:         %[[VAL_125:.*]] = load <4 x float>, ptr %[[VAL_121]], align 16
! LLVMIR:         %[[VAL_126:.*]] = load <512 x i1>, ptr %[[VAL_123]], align 64
! LLVMIR:         %[[VAL_127:.*]] = bitcast <4 x float> %[[VAL_124]] to <16 x i8>
! LLVMIR:         %[[VAL_128:.*]] = bitcast <4 x float> %[[VAL_125]] to <16 x i8>
! LLVMIR:         %[[VAL_129:.*]] = call <512 x i1> @llvm.ppc.mma.xvf32gerpn(<512 x i1> %[[VAL_126]], <16 x i8> %[[VAL_127]], <16 x i8> %[[VAL_128]])
! LLVMIR:         store <512 x i1> %[[VAL_129]], ptr %[[VAL_123]], align 64

      subroutine test_xvf32gerpp_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvf32gerpp(cq, vu10, vu11)
      end subroutine test_xvf32gerpp_u1

!CHECK-LABEL: @test_xvf32gerpp_u1_
! LLVMIR:         %[[VAL_130:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_131:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_132:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_133:.*]] = load <16 x i8>, ptr %[[VAL_131]], align 16
! LLVMIR:         %[[VAL_134:.*]] = load <16 x i8>, ptr %[[VAL_130]], align 16
! LLVMIR:         %[[VAL_135:.*]] = load <512 x i1>, ptr %[[VAL_132]], align 64
! LLVMIR:         %[[VAL_136:.*]] = call <512 x i1> @llvm.ppc.mma.xvf32gerpp(<512 x i1> %[[VAL_135]], <16 x i8> %[[VAL_133]], <16 x i8> %[[VAL_134]])
! LLVMIR:         store <512 x i1> %[[VAL_136]], ptr %[[VAL_132]], align 64


      subroutine test_xvf32gerpp_r4()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_xvf32gerpp(cq, vr40, vr41)
      end subroutine test_xvf32gerpp_r4

!CHECK-LABEL: @test_xvf32gerpp_r4_
! LLVMIR:         %[[VAL_137:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_138:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_139:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_140:.*]] = load <4 x float>, ptr %[[VAL_138]], align 16
! LLVMIR:         %[[VAL_141:.*]] = load <4 x float>, ptr %[[VAL_137]], align 16
! LLVMIR:         %[[VAL_142:.*]] = load <512 x i1>, ptr %[[VAL_139]], align 64
! LLVMIR:         %[[VAL_143:.*]] = bitcast <4 x float> %[[VAL_140]] to <16 x i8>
! LLVMIR:         %[[VAL_144:.*]] = bitcast <4 x float> %[[VAL_141]] to <16 x i8>
! LLVMIR:         %[[VAL_145:.*]] = call <512 x i1> @llvm.ppc.mma.xvf32gerpp(<512 x i1> %[[VAL_142]], <16 x i8> %[[VAL_143]], <16 x i8> %[[VAL_144]])
! LLVMIR:         store <512 x i1> %[[VAL_145]], ptr %[[VAL_139]], align 64

      subroutine test_xvf64ger_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64ger(cq, cp, vu10)
      end subroutine test_xvf64ger_u1

!CHECK-LABEL: @test_xvf64ger_u1_
! LLVMIR:         %[[VAL_146:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_147:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_148:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_149:.*]] = load <256 x i1>, ptr %[[VAL_148]], align 32
! LLVMIR:         %[[VAL_150:.*]] = load <16 x i8>, ptr %[[VAL_146]], align 16
! LLVMIR:         %[[VAL_151:.*]] = call <512 x i1> @llvm.ppc.mma.xvf64ger(<256 x i1> %[[VAL_149]], <16 x i8> %[[VAL_150]])
! LLVMIR:         store <512 x i1> %[[VAL_151]], ptr %[[VAL_147]], align 64

      subroutine test_xvf64ger_r8()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64ger(cq, cp, vr80)
      end subroutine test_xvf64ger_r8

!CHECK-LABEL: @test_xvf64ger_r8_
! LLVMIR:         %[[VAL_152:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_153:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_154:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_155:.*]] = load <256 x i1>, ptr %[[VAL_154]], align 32
! LLVMIR:         %[[VAL_156:.*]] = load <2 x double>, ptr %[[VAL_152]], align 16
! LLVMIR:         %[[VAL_157:.*]] = bitcast <2 x double> %[[VAL_156]] to <16 x i8>
! LLVMIR:         %[[VAL_158:.*]] = call <512 x i1> @llvm.ppc.mma.xvf64ger(<256 x i1> %[[VAL_155]], <16 x i8> %[[VAL_157]])
! LLVMIR:         store <512 x i1> %[[VAL_158]], ptr %[[VAL_153]], align 64


      subroutine test_xvf64gernn_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64gernn(cq, cp, vu10)
      end subroutine test_xvf64gernn_u1

!CHECK-LABEL: @test_xvf64gernn_u1_
! LLVMIR:         %[[VAL_159:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_160:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_161:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_162:.*]] = load <256 x i1>, ptr %[[VAL_161]], align 32
! LLVMIR:         %[[VAL_163:.*]] = load <16 x i8>, ptr %[[VAL_159]], align 16
! LLVMIR:         %[[VAL_164:.*]] = load <512 x i1>, ptr %[[VAL_160]], align 64
! LLVMIR:         %[[VAL_165:.*]] = call <512 x i1> @llvm.ppc.mma.xvf64gernn(<512 x i1> %[[VAL_164]], <256 x i1> %[[VAL_162]], <16 x i8> %[[VAL_163]])
! LLVMIR:         store <512 x i1> %[[VAL_165]], ptr %[[VAL_160]], align 64


      subroutine test_xvf64gernn_r8()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64gernn(cq, cp, vr80)
      end subroutine test_xvf64gernn_r8

!CHECK-LABEL: @test_xvf64gernn_r8_
! LLVMIR:         %[[VAL_166:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_167:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_168:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_169:.*]] = load <256 x i1>, ptr %[[VAL_168]], align 32
! LLVMIR:         %[[VAL_170:.*]] = load <2 x double>, ptr %[[VAL_166]], align 16
! LLVMIR:         %[[VAL_171:.*]] = load <512 x i1>, ptr %[[VAL_167]], align 64
! LLVMIR:         %[[VAL_172:.*]] = bitcast <2 x double> %[[VAL_170]] to <16 x i8>
! LLVMIR:         %[[VAL_173:.*]] = call <512 x i1> @llvm.ppc.mma.xvf64gernn(<512 x i1> %[[VAL_171]], <256 x i1> %[[VAL_169]], <16 x i8> %[[VAL_172]])
! LLVMIR:         store <512 x i1> %[[VAL_173]], ptr %[[VAL_167]], align 64

      subroutine test_xvf64gernp_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64gernp(cq, cp, vu10)
      end subroutine test_xvf64gernp_u1

!CHECK-LABEL: @test_xvf64gernp_u1_
! LLVMIR:         %[[VAL_174:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_175:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_176:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_177:.*]] = load <256 x i1>, ptr %[[VAL_176]], align 32
! LLVMIR:         %[[VAL_178:.*]] = load <16 x i8>, ptr %[[VAL_174]], align 16
! LLVMIR:         %[[VAL_179:.*]] = load <512 x i1>, ptr %[[VAL_175]], align 64
! LLVMIR:         %[[VAL_180:.*]] = call <512 x i1> @llvm.ppc.mma.xvf64gernp(<512 x i1> %[[VAL_179]], <256 x i1> %[[VAL_177]], <16 x i8> %[[VAL_178]])
! LLVMIR:         store <512 x i1> %[[VAL_180]], ptr %[[VAL_175]], align 64

      subroutine test_xvf64gernp_r8()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vr80
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64gernp(cq, cp, vr80)
      end subroutine test_xvf64gernp_r8

!CHECK-LABEL: @test_xvf64gernp_r8_
! LLVMIR:         %[[VAL_181:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_182:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_183:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_184:.*]] = load <256 x i1>, ptr %[[VAL_183]], align 32
! LLVMIR:         %[[VAL_185:.*]] = load <16 x i8>, ptr %[[VAL_181]], align 16
! LLVMIR:         %[[VAL_186:.*]] = load <512 x i1>, ptr %[[VAL_182]], align 64
! LLVMIR:         %[[VAL_187:.*]] = call <512 x i1> @llvm.ppc.mma.xvf64gernp(<512 x i1> %[[VAL_186]], <256 x i1> %[[VAL_184]], <16 x i8> %[[VAL_185]])
! LLVMIR:         store <512 x i1> %[[VAL_187]], ptr %[[VAL_182]], align 64

      subroutine test_xvf64gerpn_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64gerpn(cq, cp, vu10)
      end subroutine test_xvf64gerpn_u1

!CHECK-LABEL: @test_xvf64gerpn_u1_
! LLVMIR:         %[[VAL_188:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_189:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_190:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_191:.*]] = load <256 x i1>, ptr %[[VAL_190]], align 32
! LLVMIR:         %[[VAL_192:.*]] = load <16 x i8>, ptr %[[VAL_188]], align 16
! LLVMIR:         %[[VAL_193:.*]] = load <512 x i1>, ptr %[[VAL_189]], align 64
! LLVMIR:         %[[VAL_194:.*]] = call <512 x i1> @llvm.ppc.mma.xvf64gerpn(<512 x i1> %[[VAL_193]], <256 x i1> %[[VAL_191]], <16 x i8> %[[VAL_192]])
! LLVMIR:         store <512 x i1> %[[VAL_194]], ptr %[[VAL_189]], align 64

      subroutine test_xvf64gerpn_r8()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64gerpn(cq, cp, vr80)
      end subroutine test_xvf64gerpn_r8

!CHECK-LABEL: @test_xvf64gerpn_r8_
! LLVMIR:         %[[VAL_195:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_196:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_197:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_198:.*]] = load <256 x i1>, ptr %[[VAL_197]], align 32
! LLVMIR:         %[[VAL_199:.*]] = load <2 x double>, ptr %[[VAL_195]], align 16
! LLVMIR:         %[[VAL_200:.*]] = load <512 x i1>, ptr %[[VAL_196]], align 64
! LLVMIR:         %[[VAL_201:.*]] = bitcast <2 x double> %[[VAL_199]] to <16 x i8>
! LLVMIR:         %[[VAL_202:.*]] = call <512 x i1> @llvm.ppc.mma.xvf64gerpn(<512 x i1> %[[VAL_200]], <256 x i1> %[[VAL_198]], <16 x i8> %[[VAL_201]])
! LLVMIR:         store <512 x i1> %[[VAL_202]], ptr %[[VAL_196]], align 64

      subroutine test_xvf64gerpp_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64gerpp(cq, cp, vu10)
      end subroutine test_xvf64gerpp_u1

!CHECK-LABEL: @test_xvf64gerpp_u1_
! LLVMIR:         %[[VAL_203:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_204:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_205:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_206:.*]] = load <256 x i1>, ptr %[[VAL_205]], align 32
! LLVMIR:         %[[VAL_207:.*]] = load <16 x i8>, ptr %[[VAL_203]], align 16
! LLVMIR:         %[[VAL_208:.*]] = load <512 x i1>, ptr %[[VAL_204]], align 64
! LLVMIR:         %[[VAL_209:.*]] = call <512 x i1> @llvm.ppc.mma.xvf64gerpp(<512 x i1> %[[VAL_208]], <256 x i1> %[[VAL_206]], <16 x i8> %[[VAL_207]])
! LLVMIR:         store <512 x i1> %[[VAL_209]], ptr %[[VAL_204]], align 64


      subroutine test_xvf64gerpp_r8()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_quad :: cq
      __vector_pair :: cp
      call mma_xvf64gerpp(cq, cp, vr80)
      end subroutine test_xvf64gerpp_r8

!CHECK-LABEL: @test_xvf64gerpp_r8_
! LLVMIR:         %[[VAL_210:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_211:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_212:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_213:.*]] = load <256 x i1>, ptr %[[VAL_212]], align 32
! LLVMIR:         %[[VAL_214:.*]] = load <2 x double>, ptr %[[VAL_210]], align 16
! LLVMIR:         %[[VAL_215:.*]] = load <512 x i1>, ptr %[[VAL_211]], align 64
! LLVMIR:         %[[VAL_216:.*]] = bitcast <2 x double> %[[VAL_214]] to <16 x i8>
! LLVMIR:         %[[VAL_217:.*]] = call <512 x i1> @llvm.ppc.mma.xvf64gerpp(<512 x i1> %[[VAL_215]], <256 x i1> %[[VAL_213]], <16 x i8> %[[VAL_216]])
! LLVMIR:         store <512 x i1> %[[VAL_217]], ptr %[[VAL_211]], align 64

      subroutine test_xvi16ger2_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi16ger2(cq, vu10, vu11)
      end subroutine test_xvi16ger2_u1

!CHECK-LABEL: @test_xvi16ger2_u1_
! LLVMIR:         %[[VAL_218:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_219:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_220:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_221:.*]] = load <16 x i8>, ptr %[[VAL_219]], align 16
! LLVMIR:         %[[VAL_222:.*]] = load <16 x i8>, ptr %[[VAL_218]], align 16
! LLVMIR:         %[[VAL_223:.*]] = call <512 x i1> @llvm.ppc.mma.xvi16ger2(<16 x i8> %[[VAL_221]], <16 x i8> %[[VAL_222]])
! LLVMIR:         store <512 x i1> %[[VAL_223]], ptr %[[VAL_220]], align 64

      subroutine test_xvi16ger2_i2()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_xvi16ger2(cq, vi20, vi21)
      end subroutine test_xvi16ger2_i2

!CHECK-LABEL: @test_xvi16ger2_i2_
! LLVMIR:         %[[VAL_224:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_225:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_226:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_227:.*]] = load <8 x i16>, ptr %[[VAL_225]], align 16
! LLVMIR:         %[[VAL_228:.*]] = load <8 x i16>, ptr %[[VAL_224]], align 16
! LLVMIR:         %[[VAL_229:.*]] = bitcast <8 x i16> %[[VAL_227]] to <16 x i8>
! LLVMIR:         %[[VAL_230:.*]] = bitcast <8 x i16> %[[VAL_228]] to <16 x i8>
! LLVMIR:         %[[VAL_231:.*]] = call <512 x i1> @llvm.ppc.mma.xvi16ger2(<16 x i8> %[[VAL_229]], <16 x i8> %[[VAL_230]])
! LLVMIR:         store <512 x i1> %[[VAL_231]], ptr %[[VAL_226]], align 64

      subroutine test_xvi16ger2pp_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi16ger2pp(cq, vu10, vu11)
      end subroutine test_xvi16ger2pp_u1

!CHECK-LABEL: @test_xvi16ger2pp_u1_
! LLVMIR:         %[[VAL_232:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_233:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_234:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_235:.*]] = load <16 x i8>, ptr %[[VAL_233]], align 16
! LLVMIR:         %[[VAL_236:.*]] = load <16 x i8>, ptr %[[VAL_232]], align 16
! LLVMIR:         %[[VAL_237:.*]] = load <512 x i1>, ptr %[[VAL_234]], align 64
! LLVMIR:         %[[VAL_238:.*]] = call <512 x i1> @llvm.ppc.mma.xvi16ger2pp(<512 x i1> %[[VAL_237]], <16 x i8> %[[VAL_235]], <16 x i8> %[[VAL_236]])
! LLVMIR:         store <512 x i1> %[[VAL_238]], ptr %[[VAL_234]], align 64

      subroutine test_xvi16ger2pp_i2()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_xvi16ger2pp(cq, vi20, vi21)
      end subroutine test_xvi16ger2pp_i2

!CHECK-LABEL: @test_xvi16ger2pp_i2_
! LLVMIR:         %[[VAL_239:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_240:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_241:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_242:.*]] = load <8 x i16>, ptr %[[VAL_240]], align 16
! LLVMIR:         %[[VAL_243:.*]] = load <8 x i16>, ptr %[[VAL_239]], align 16
! LLVMIR:         %[[VAL_244:.*]] = load <512 x i1>, ptr %[[VAL_241]], align 64
! LLVMIR:         %[[VAL_245:.*]] = bitcast <8 x i16> %[[VAL_242]] to <16 x i8>
! LLVMIR:         %[[VAL_246:.*]] = bitcast <8 x i16> %[[VAL_243]] to <16 x i8>
! LLVMIR:         %[[VAL_247:.*]] = call <512 x i1> @llvm.ppc.mma.xvi16ger2pp(<512 x i1> %[[VAL_244]], <16 x i8> %[[VAL_245]], <16 x i8> %[[VAL_246]])
! LLVMIR:         store <512 x i1> %[[VAL_247]], ptr %[[VAL_241]], align 64

      subroutine test_xvi16ger2s_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi16ger2s(cq, vu10, vu11)
      end subroutine test_xvi16ger2s_u1

!CHECK-LABEL:  @test_xvi16ger2s_u1_
! LLVMIR:         %[[VAL_248:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_249:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_250:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_251:.*]] = load <16 x i8>, ptr %[[VAL_249]], align 16
! LLVMIR:         %[[VAL_252:.*]] = load <16 x i8>, ptr %[[VAL_248]], align 16
! LLVMIR:         %[[VAL_253:.*]] = call <512 x i1> @llvm.ppc.mma.xvi16ger2s(<16 x i8> %[[VAL_251]], <16 x i8> %[[VAL_252]])
! LLVMIR:         store <512 x i1> %[[VAL_253]], ptr %[[VAL_250]], align 64

      subroutine test_xvi16ger2s_i2()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_xvi16ger2s(cq, vi20, vi21)
      end subroutine test_xvi16ger2s_i2

!CHECK-LABEL:  @test_xvi16ger2s_i2_
! LLVMIR:         %[[VAL_254:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_255:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_256:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_257:.*]] = load <8 x i16>, ptr %[[VAL_255]], align 16
! LLVMIR:         %[[VAL_258:.*]] = load <8 x i16>, ptr %[[VAL_254]], align 16
! LLVMIR:         %[[VAL_259:.*]] = bitcast <8 x i16> %[[VAL_257]] to <16 x i8>
! LLVMIR:         %[[VAL_260:.*]] = bitcast <8 x i16> %[[VAL_258]] to <16 x i8>
! LLVMIR:         %[[VAL_261:.*]] = call <512 x i1> @llvm.ppc.mma.xvi16ger2s(<16 x i8> %[[VAL_259]], <16 x i8> %[[VAL_260]])
! LLVMIR:         store <512 x i1> %[[VAL_261]], ptr %[[VAL_256]], align 64

      subroutine test_xvi16ger2spp_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi16ger2spp(cq, vu10, vu11)
      end subroutine test_xvi16ger2spp_u1

!CHECK-LABEL:  @test_xvi16ger2spp_u1_
! LLVMIR:         %[[VAL_262:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_263:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_264:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_265:.*]] = load <16 x i8>, ptr %[[VAL_263]], align 16
! LLVMIR:         %[[VAL_266:.*]] = load <16 x i8>, ptr %[[VAL_262]], align 16
! LLVMIR:         %[[VAL_267:.*]] = load <512 x i1>, ptr %[[VAL_264]], align 64
! LLVMIR:         %[[VAL_268:.*]] = call <512 x i1> @llvm.ppc.mma.xvi16ger2spp(<512 x i1> %[[VAL_267]], <16 x i8> %[[VAL_265]], <16 x i8> %[[VAL_266]])
! LLVMIR:         store <512 x i1> %[[VAL_268]], ptr %[[VAL_264]], align 64

      subroutine test_xvi16ger2spp_i2()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_xvi16ger2spp(cq, vi20, vi21)
      end subroutine test_xvi16ger2spp_i2

!CHECK-LABEL:  @test_xvi16ger2spp_i2_
! LLVMIR:         %[[VAL_269:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_270:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_271:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_272:.*]] = load <8 x i16>, ptr %[[VAL_270]], align 16
! LLVMIR:         %[[VAL_273:.*]] = load <8 x i16>, ptr %[[VAL_269]], align 16
! LLVMIR:         %[[VAL_274:.*]] = load <512 x i1>, ptr %[[VAL_271]], align 64
! LLVMIR:         %[[VAL_275:.*]] = bitcast <8 x i16> %[[VAL_272]] to <16 x i8>
! LLVMIR:         %[[VAL_276:.*]] = bitcast <8 x i16> %[[VAL_273]] to <16 x i8>
! LLVMIR:         %[[VAL_277:.*]] = call <512 x i1> @llvm.ppc.mma.xvi16ger2spp(<512 x i1> %[[VAL_274]], <16 x i8> %[[VAL_275]], <16 x i8> %[[VAL_276]])
! LLVMIR:         store <512 x i1> %[[VAL_277]], ptr %[[VAL_271]], align 64

      subroutine test_xvi4ger8()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi4ger8(cq, vu10, vu11)
      end subroutine test_xvi4ger8

!CHECK-LABEL:  @test_xvi4ger8_
! LLVMIR:         %[[VAL_278:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_279:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_280:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_281:.*]] = load <16 x i8>, ptr %[[VAL_279]], align 16
! LLVMIR:         %[[VAL_282:.*]] = load <16 x i8>, ptr %[[VAL_278]], align 16
! LLVMIR:         %[[VAL_283:.*]] = call <512 x i1> @llvm.ppc.mma.xvi4ger8(<16 x i8> %[[VAL_281]], <16 x i8> %[[VAL_282]])
! LLVMIR:         store <512 x i1> %[[VAL_283]], ptr %[[VAL_280]], align 64

      subroutine test_xvi4ger8pp()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi4ger8pp(cq, vu10, vu11)
      end subroutine test_xvi4ger8pp

!CHECK-LABEL:  @test_xvi4ger8pp_
! LLVMIR:         %[[VAL_284:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_285:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_286:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_287:.*]] = load <16 x i8>, ptr %[[VAL_285]], align 16
! LLVMIR:         %[[VAL_288:.*]] = load <16 x i8>, ptr %[[VAL_284]], align 16
! LLVMIR:         %[[VAL_289:.*]] = load <512 x i1>, ptr %[[VAL_286]], align 64
! LLVMIR:         %[[VAL_290:.*]] = call <512 x i1> @llvm.ppc.mma.xvi4ger8pp(<512 x i1> %[[VAL_289]], <16 x i8> %[[VAL_287]], <16 x i8> %[[VAL_288]])
! LLVMIR:         store <512 x i1> %[[VAL_290]], ptr %[[VAL_286]], align 64

      subroutine test_xvi8ger4_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi8ger4(cq, vu10, vu11)
      end subroutine test_xvi8ger4_u1

!CHECK-LABEL: @test_xvi8ger4_u1_
! LLVMIR:         %[[VAL_291:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_292:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_293:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_294:.*]] = load <16 x i8>, ptr %[[VAL_292]], align 16
! LLVMIR:         %[[VAL_295:.*]] = load <16 x i8>, ptr %[[VAL_291]], align 16
! LLVMIR:         %[[VAL_296:.*]] = call <512 x i1> @llvm.ppc.mma.xvi8ger4(<16 x i8> %[[VAL_294]], <16 x i8> %[[VAL_295]])
! LLVMIR:         store <512 x i1> %[[VAL_296]], ptr %[[VAL_293]], align 64


      subroutine test_xvi8ger4_i1()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_xvi8ger4(cq, vi10, vi11)
      end subroutine test_xvi8ger4_i1

!CHECK-LABEL: @test_xvi8ger4_i1_
! LLVMIR:         %[[VAL_297:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_298:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_299:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_300:.*]] = load <16 x i8>, ptr %[[VAL_298]], align 16
! LLVMIR:         %[[VAL_301:.*]] = load <16 x i8>, ptr %[[VAL_297]], align 16
! LLVMIR:         %[[VAL_302:.*]] = call <512 x i1> @llvm.ppc.mma.xvi8ger4(<16 x i8> %[[VAL_300]], <16 x i8> %[[VAL_301]])
! LLVMIR:         store <512 x i1> %[[VAL_302]], ptr %[[VAL_299]], align 64

      subroutine test_xvi8ger4pp_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi8ger4pp(cq, vu10, vu11)
      end subroutine test_xvi8ger4pp_u1

!CHECK-LABEL: @test_xvi8ger4pp_u1_
! LLVMIR:         %[[VAL_303:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_304:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_305:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_306:.*]] = load <16 x i8>, ptr %[[VAL_304]], align 16
! LLVMIR:         %[[VAL_307:.*]] = load <16 x i8>, ptr %[[VAL_303]], align 16
! LLVMIR:         %[[VAL_308:.*]] = load <512 x i1>, ptr %[[VAL_305]], align 64
! LLVMIR:         %[[VAL_309:.*]] = call <512 x i1> @llvm.ppc.mma.xvi8ger4pp(<512 x i1> %[[VAL_308]], <16 x i8> %[[VAL_306]], <16 x i8> %[[VAL_307]])
! LLVMIR:         store <512 x i1> %[[VAL_309]], ptr %[[VAL_305]], align 64

      subroutine test_xvi8ger4pp_i1()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_xvi8ger4pp(cq, vi10, vi11)
      end subroutine test_xvi8ger4pp_i1

!CHECK-LABEL:  @test_xvi8ger4pp_i1_
! LLVMIR:         %[[VAL_310:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_311:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_312:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_313:.*]] = load <16 x i8>, ptr %[[VAL_311]], align 16
! LLVMIR:         %[[VAL_314:.*]] = load <16 x i8>, ptr %[[VAL_310]], align 16
! LLVMIR:         %[[VAL_315:.*]] = load <512 x i1>, ptr %[[VAL_312]], align 64
! LLVMIR:         %[[VAL_316:.*]] = call <512 x i1> @llvm.ppc.mma.xvi8ger4pp(<512 x i1> %[[VAL_315]], <16 x i8> %[[VAL_313]], <16 x i8> %[[VAL_314]])
! LLVMIR:         store <512 x i1> %[[VAL_316]], ptr %[[VAL_312]], align 64

      subroutine test_xvi8ger4spp_u1()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_xvi8ger4spp(cq, vu10, vu11)
      end subroutine test_xvi8ger4spp_u1

!CHECK-LABEL: @test_xvi8ger4spp_u1_
! LLVMIR:         %[[VAL_317:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_318:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_319:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_320:.*]] = load <16 x i8>, ptr %[[VAL_318]], align 16
! LLVMIR:         %[[VAL_321:.*]] = load <16 x i8>, ptr %[[VAL_317]], align 16
! LLVMIR:         %[[VAL_322:.*]] = load <512 x i1>, ptr %[[VAL_319]], align 64
! LLVMIR:         %[[VAL_323:.*]] = call <512 x i1> @llvm.ppc.mma.xvi8ger4spp(<512 x i1> %[[VAL_322]], <16 x i8> %[[VAL_320]], <16 x i8> %[[VAL_321]])
! LLVMIR:         store <512 x i1> %[[VAL_323]], ptr %[[VAL_319]], align 64

      subroutine test_xvi8ger4spp_i1()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_xvi8ger4spp(cq, vi10, vi11)
      end subroutine test_xvi8ger4spp_i1

!CHECK-LABEL: @test_xvi8ger4spp_i1_
! LLVMIR:         %[[VAL_324:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_325:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_326:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_327:.*]] = load <16 x i8>, ptr %[[VAL_325]], align 16
! LLVMIR:         %[[VAL_328:.*]] = load <16 x i8>, ptr %[[VAL_324]], align 16
! LLVMIR:         %[[VAL_329:.*]] = load <512 x i1>, ptr %[[VAL_326]], align 64
! LLVMIR:         %[[VAL_330:.*]] = call <512 x i1> @llvm.ppc.mma.xvi8ger4spp(<512 x i1> %[[VAL_329]], <16 x i8> %[[VAL_327]], <16 x i8> %[[VAL_328]])
! LLVMIR:         store <512 x i1> %[[VAL_330]], ptr %[[VAL_326]], align 64
