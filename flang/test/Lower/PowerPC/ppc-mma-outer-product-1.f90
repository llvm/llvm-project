! RUN: %flang_fc1 -flang-experimental-hlfir -triple powerpc64le-unknown-unknown -target-cpu pwr10 -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

      subroutine test_pmxvbf16ger2_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvbf16ger2_def

!CHECK-LABEL: @test_pmxvbf16ger2_def_
! LLVMIR:         %[[VAL_0:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_1:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_2:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_3:.*]] = load <16 x i8>, ptr %[[VAL_1]], align 16
! LLVMIR:         %[[VAL_4:.*]] = load <16 x i8>, ptr %[[VAL_0]], align 16
! LLVMIR:         %[[VAL_5:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2(<16 x i8> %[[VAL_3]], <16 x i8> %[[VAL_4]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_5]], ptr %[[VAL_2]], align 64


      subroutine test_pmxvbf16ger2_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvbf16ger2_non_def

!CHECK-LABEL: @test_pmxvbf16ger2_non_def_
! LLVMIR:         %[[VAL_6:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_7:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_8:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_9:.*]] = load <16 x i8>, ptr %[[VAL_7]], align 16
! LLVMIR:         %[[VAL_10:.*]] = load <16 x i8>, ptr %[[VAL_6]], align 16
! LLVMIR:         %[[VAL_11:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2(<16 x i8> %[[VAL_9]], <16 x i8> %[[VAL_10]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_11]], ptr %[[VAL_8]], align 64


      subroutine test_pmxvbf16ger2nn_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2nn(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvbf16ger2nn_def

!CHECK-LABEL: @test_pmxvbf16ger2nn_def_
! LLVMIR:         %[[VAL_12:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_13:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_14:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_15:.*]] = load <16 x i8>, ptr %[[VAL_13]], align 16
! LLVMIR:         %[[VAL_16:.*]] = load <16 x i8>, ptr %[[VAL_12]], align 16
! LLVMIR:         %[[VAL_17:.*]] = load <512 x i1>, ptr %[[VAL_14]], align 64
! LLVMIR:         %[[VAL_18:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2nn(<512 x i1> %[[VAL_17]], <16 x i8> %[[VAL_15]], <16 x i8> %[[VAL_16]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_18]], ptr %[[VAL_14]], align 64

      subroutine test_pmxvbf16ger2nn_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2nn(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvbf16ger2nn_non_def

!CHECK-LABEL: @test_pmxvbf16ger2nn_non_def_
! LLVMIR:         %[[VAL_19:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_20:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_21:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_22:.*]] = load <16 x i8>, ptr %[[VAL_20]], align 16
! LLVMIR:         %[[VAL_23:.*]] = load <16 x i8>, ptr %[[VAL_19]], align 16
! LLVMIR:         %[[VAL_24:.*]] = load <512 x i1>, ptr %[[VAL_21]], align 64
! LLVMIR:         %[[VAL_25:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2nn(<512 x i1> %[[VAL_24]], <16 x i8> %[[VAL_22]], <16 x i8> %[[VAL_23]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_25]], ptr %[[VAL_21]], align 64

      subroutine test_pmxvbf16ger2np_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2np(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvbf16ger2np_def

!CHECK-LABEL: @test_pmxvbf16ger2np_def_
! LLVMIR:         %[[VAL_26:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_27:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_28:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_29:.*]] = load <16 x i8>, ptr %[[VAL_27]], align 16
! LLVMIR:         %[[VAL_30:.*]] = load <16 x i8>, ptr %[[VAL_26]], align 16
! LLVMIR:         %[[VAL_31:.*]] = load <512 x i1>, ptr %[[VAL_28]], align 64
! LLVMIR:         %[[VAL_32:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2np(<512 x i1> %[[VAL_31]], <16 x i8> %[[VAL_29]], <16 x i8> %[[VAL_30]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_32]], ptr %[[VAL_28]], align 64

      subroutine test_pmxvbf16ger2np_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2np(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvbf16ger2np_non_def

!CHECK-LABEL: @test_pmxvbf16ger2np_non_def_
! LLVMIR:         %[[VAL_33:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_34:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_35:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_36:.*]] = load <16 x i8>, ptr %[[VAL_34]], align 16
! LLVMIR:         %[[VAL_37:.*]] = load <16 x i8>, ptr %[[VAL_33]], align 16
! LLVMIR:         %[[VAL_38:.*]] = load <512 x i1>, ptr %[[VAL_35]], align 64
! LLVMIR:         %[[VAL_39:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2np(<512 x i1> %[[VAL_38]], <16 x i8> %[[VAL_36]], <16 x i8> %[[VAL_37]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_39]], ptr %[[VAL_35]], align 64

      subroutine test_pmxvbf16ger2pn_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2pn(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvbf16ger2pn_def

!CHECK-LABEL: @test_pmxvbf16ger2pn_def_
! LLVMIR:         %[[VAL_40:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_41:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_42:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_43:.*]] = load <16 x i8>, ptr %[[VAL_41]], align 16
! LLVMIR:         %[[VAL_44:.*]] = load <16 x i8>, ptr %[[VAL_40]], align 16
! LLVMIR:         %[[VAL_45:.*]] = load <512 x i1>, ptr %[[VAL_42]], align 64
! LLVMIR:         %[[VAL_46:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2pn(<512 x i1> %[[VAL_45]], <16 x i8> %[[VAL_43]], <16 x i8> %[[VAL_44]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_46]], ptr %[[VAL_42]], align 64

      subroutine test_pmxvbf16ger2pn_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2pn(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvbf16ger2pn_non_def

!CHECK-LABEL: @test_pmxvbf16ger2pn_non_def_
! LLVMIR:         %[[VAL_47:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_48:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_49:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_50:.*]] = load <16 x i8>, ptr %[[VAL_48]], align 16
! LLVMIR:         %[[VAL_51:.*]] = load <16 x i8>, ptr %[[VAL_47]], align 16
! LLVMIR:         %[[VAL_52:.*]] = load <512 x i1>, ptr %[[VAL_49]], align 64
! LLVMIR:         %[[VAL_53:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2pn(<512 x i1> %[[VAL_52]], <16 x i8> %[[VAL_50]], <16 x i8> %[[VAL_51]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_53]], ptr %[[VAL_49]], align 64

      subroutine test_pmxvbf16ger2pp_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2pp(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvbf16ger2pp_def

!CHECK-LABEL: @test_pmxvbf16ger2pp_def_
! LLVMIR:         %[[VAL_54:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_55:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_56:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_57:.*]] = load <16 x i8>, ptr %[[VAL_55]], align 16
! LLVMIR:         %[[VAL_58:.*]] = load <16 x i8>, ptr %[[VAL_54]], align 16
! LLVMIR:         %[[VAL_59:.*]] = load <512 x i1>, ptr %[[VAL_56]], align 64
! LLVMIR:         %[[VAL_60:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2pp(<512 x i1> %[[VAL_59]], <16 x i8> %[[VAL_57]], <16 x i8> %[[VAL_58]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_60]], ptr %[[VAL_56]], align 64

      subroutine test_pmxvbf16ger2pp_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvbf16ger2pp(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvbf16ger2pp_non_def

!CHECK-LABEL: @test_pmxvbf16ger2pp_non_def_
! LLVMIR:         %[[VAL_61:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_62:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_63:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_64:.*]] = load <16 x i8>, ptr %[[VAL_62]], align 16
! LLVMIR:         %[[VAL_65:.*]] = load <16 x i8>, ptr %[[VAL_61]], align 16
! LLVMIR:         %[[VAL_66:.*]] = load <512 x i1>, ptr %[[VAL_63]], align 64
! LLVMIR:         %[[VAL_67:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvbf16ger2pp(<512 x i1> %[[VAL_66]], <16 x i8> %[[VAL_64]], <16 x i8> %[[VAL_65]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_67]], ptr %[[VAL_63]], align 64

      subroutine test_pmxvf16ger2_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvf16ger2_def

!CHECK-LABEL: @test_pmxvf16ger2_def_
! LLVMIR:         %[[VAL_68:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_69:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_70:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_71:.*]] = load <16 x i8>, ptr %[[VAL_69]], align 16
! LLVMIR:         %[[VAL_72:.*]] = load <16 x i8>, ptr %[[VAL_68]], align 16
! LLVMIR:         %[[VAL_73:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2(<16 x i8> %[[VAL_71]], <16 x i8> %[[VAL_72]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_73]], ptr %[[VAL_70]], align 64

      subroutine test_pmxvf16ger2_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvf16ger2_non_def

!CHECK-LABEL: @test_pmxvf16ger2_non_def_
! LLVMIR:         %[[VAL_74:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_75:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_76:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_77:.*]] = load <16 x i8>, ptr %[[VAL_75]], align 16
! LLVMIR:         %[[VAL_78:.*]] = load <16 x i8>, ptr %[[VAL_74]], align 16
! LLVMIR:         %[[VAL_79:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2(<16 x i8> %[[VAL_77]], <16 x i8> %[[VAL_78]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_79]], ptr %[[VAL_76]], align 64

      subroutine test_pmxvf16ger2nn_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2nn(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvf16ger2nn_def

!CHECK-LABEL: @test_pmxvf16ger2nn_def_
! LLVMIR:         %[[VAL_80:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_81:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_82:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_83:.*]] = load <16 x i8>, ptr %[[VAL_81]], align 16
! LLVMIR:         %[[VAL_84:.*]] = load <16 x i8>, ptr %[[VAL_80]], align 16
! LLVMIR:         %[[VAL_85:.*]] = load <512 x i1>, ptr %[[VAL_82]], align 64
! LLVMIR:         %[[VAL_86:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2nn(<512 x i1> %[[VAL_85]], <16 x i8> %[[VAL_83]], <16 x i8> %[[VAL_84]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_86]], ptr %[[VAL_82]], align 64

      subroutine test_pmxvf16ger2nn_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2nn(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvf16ger2nn_non_def

!CHECK-LABEL: @test_pmxvf16ger2nn_non_def_
! LLVMIR:         %[[VAL_87:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_88:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_89:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_90:.*]] = load <16 x i8>, ptr %[[VAL_88]], align 16
! LLVMIR:         %[[VAL_91:.*]] = load <16 x i8>, ptr %[[VAL_87]], align 16
! LLVMIR:         %[[VAL_92:.*]] = load <512 x i1>, ptr %[[VAL_89]], align 64
! LLVMIR:         %[[VAL_93:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2nn(<512 x i1> %[[VAL_92]], <16 x i8> %[[VAL_90]], <16 x i8> %[[VAL_91]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_93]], ptr %[[VAL_89]], align 64

      subroutine test_pmxvf16ger2np_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2np(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvf16ger2np_def

!CHECK-LABEL: @test_pmxvf16ger2np_def_
! LLVMIR:         %[[VAL_94:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_95:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_96:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_97:.*]] = load <16 x i8>, ptr %[[VAL_95]], align 16
! LLVMIR:         %[[VAL_98:.*]] = load <16 x i8>, ptr %[[VAL_94]], align 16
! LLVMIR:         %[[VAL_99:.*]] = load <512 x i1>, ptr %[[VAL_96]], align 64
! LLVMIR:         %[[VAL_100:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2np(<512 x i1> %[[VAL_99]], <16 x i8> %[[VAL_97]], <16 x i8> %[[VAL_98]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_100]], ptr %[[VAL_96]], align 64

      subroutine test_pmxvf16ger2np_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2np(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvf16ger2np_non_def

!CHECK-LABEL: @test_pmxvf16ger2np_non_def_
! LLVMIR:         %[[VAL_101:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_102:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_103:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_104:.*]] = load <16 x i8>, ptr %[[VAL_102]], align 16
! LLVMIR:         %[[VAL_105:.*]] = load <16 x i8>, ptr %[[VAL_101]], align 16
! LLVMIR:         %[[VAL_106:.*]] = load <512 x i1>, ptr %[[VAL_103]], align 64
! LLVMIR:         %[[VAL_107:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2np(<512 x i1> %[[VAL_106]], <16 x i8> %[[VAL_104]], <16 x i8> %[[VAL_105]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_107]], ptr %[[VAL_103]], align 64

      subroutine test_pmxvf16ger2pn_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2pn(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvf16ger2pn_def

!CHECK-LABEL: @test_pmxvf16ger2pn_def_
! LLVMIR:         %[[VAL_108:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_109:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_110:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_111:.*]] = load <16 x i8>, ptr %[[VAL_109]], align 16
! LLVMIR:         %[[VAL_112:.*]] = load <16 x i8>, ptr %[[VAL_108]], align 16
! LLVMIR:         %[[VAL_113:.*]] = load <512 x i1>, ptr %[[VAL_110]], align 64
! LLVMIR:         %[[VAL_114:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2pn(<512 x i1> %[[VAL_113]], <16 x i8> %[[VAL_111]], <16 x i8> %[[VAL_112]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_114]], ptr %[[VAL_110]], align 64

      subroutine test_pmxvf16ger2pn_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2pn(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvf16ger2pn_non_def

!CHECK-LABEL: @test_pmxvf16ger2pn_non_def_
! LLVMIR:         %[[VAL_115:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_116:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_117:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_118:.*]] = load <16 x i8>, ptr %[[VAL_116]], align 16
! LLVMIR:         %[[VAL_119:.*]] = load <16 x i8>, ptr %[[VAL_115]], align 16
! LLVMIR:         %[[VAL_120:.*]] = load <512 x i1>, ptr %[[VAL_117]], align 64
! LLVMIR:         %[[VAL_121:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2pn(<512 x i1> %[[VAL_120]], <16 x i8> %[[VAL_118]], <16 x i8> %[[VAL_119]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_121]], ptr %[[VAL_117]], align 64

      subroutine test_pmxvf16ger2pp_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2pp(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvf16ger2pp_def

!CHECK-LABEL: @test_pmxvf16ger2pp_def_
! LLVMIR:         %[[VAL_122:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_123:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_124:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_125:.*]] = load <16 x i8>, ptr %[[VAL_123]], align 16
! LLVMIR:         %[[VAL_126:.*]] = load <16 x i8>, ptr %[[VAL_122]], align 16
! LLVMIR:         %[[VAL_127:.*]] = load <512 x i1>, ptr %[[VAL_124]], align 64
! LLVMIR:         %[[VAL_128:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2pp(<512 x i1> %[[VAL_127]], <16 x i8> %[[VAL_125]], <16 x i8> %[[VAL_126]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_128]], ptr %[[VAL_124]], align 64

      subroutine test_pmxvf16ger2pp_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf16ger2pp(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvf16ger2pp_non_def

!CHECK-LABEL: @test_pmxvf16ger2pp_non_def_
! LLVMIR:         %[[VAL_129:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_130:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_131:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_132:.*]] = load <16 x i8>, ptr %[[VAL_130]], align 16
! LLVMIR:         %[[VAL_133:.*]] = load <16 x i8>, ptr %[[VAL_129]], align 16
! LLVMIR:         %[[VAL_134:.*]] = load <512 x i1>, ptr %[[VAL_131]], align 64
! LLVMIR:         %[[VAL_135:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf16ger2pp(<512 x i1> %[[VAL_134]], <16 x i8> %[[VAL_132]], <16 x i8> %[[VAL_133]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_135]], ptr %[[VAL_131]], align 64

      subroutine test_pmxvf32ger_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32ger(cq, vu10, vu11, 7, 2)
      end subroutine test_pmxvf32ger_u1_def

!CHECK-LABEL: @test_pmxvf32ger_u1_def_
! LLVMIR:         %[[VAL_136:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_137:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_138:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_139:.*]] = load <16 x i8>, ptr %[[VAL_137]], align 16
! LLVMIR:         %[[VAL_140:.*]] = load <16 x i8>, ptr %[[VAL_136]], align 16
! LLVMIR:         %[[VAL_141:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32ger(<16 x i8> %[[VAL_139]], <16 x i8> %[[VAL_140]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_141]], ptr %[[VAL_138]], align 64

      subroutine test_pmxvf32ger_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32ger(cq, vu10, vu11, 7_2, 2_1)
      end subroutine test_pmxvf32ger_u1_non_def

!CHECK-LABEL: @test_pmxvf32ger_u1_non_def_
! LLVMIR:         %[[VAL_142:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_143:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_144:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_145:.*]] = load <16 x i8>, ptr %[[VAL_143]], align 16
! LLVMIR:         %[[VAL_146:.*]] = load <16 x i8>, ptr %[[VAL_142]], align 16
! LLVMIR:         %[[VAL_147:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32ger(<16 x i8> %[[VAL_145]], <16 x i8> %[[VAL_146]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_147]], ptr %[[VAL_144]], align 64

      subroutine test_pmxvf32ger_r4_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32ger(cq, vr40, vr41, 7, 2)
      end subroutine test_pmxvf32ger_r4_def

!CHECK-LABEL: @test_pmxvf32ger_r4_def_
! LLVMIR:         %[[VAL_148:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_149:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_150:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_151:.*]] = load <4 x float>, ptr %[[VAL_149]], align 16
! LLVMIR:         %[[VAL_152:.*]] = load <4 x float>, ptr %[[VAL_148]], align 16
! LLVMIR:         %[[VAL_153:.*]] = bitcast <4 x float> %[[VAL_151]] to <16 x i8>
! LLVMIR:         %[[VAL_154:.*]] = bitcast <4 x float> %[[VAL_152]] to <16 x i8>
! LLVMIR:         %[[VAL_155:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32ger(<16 x i8> %[[VAL_153]], <16 x i8> %[[VAL_154]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_155]], ptr %[[VAL_150]], align 64

      subroutine test_pmxvf32ger_r4_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32ger(cq, vr40, vr41, 7_2, 2_1)
      end subroutine test_pmxvf32ger_r4_non_def

!CHECK-LABEL: @test_pmxvf32ger_r4_non_def_
! LLVMIR:         %[[VAL_156:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_157:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_158:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_159:.*]] = load <4 x float>, ptr %[[VAL_157]], align 16
! LLVMIR:         %[[VAL_160:.*]] = load <4 x float>, ptr %[[VAL_156]], align 16
! LLVMIR:         %[[VAL_161:.*]] = bitcast <4 x float> %[[VAL_159]] to <16 x i8>
! LLVMIR:         %[[VAL_162:.*]] = bitcast <4 x float> %[[VAL_160]] to <16 x i8>
! LLVMIR:         %[[VAL_163:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32ger(<16 x i8> %[[VAL_161]], <16 x i8> %[[VAL_162]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_163]], ptr %[[VAL_158]], align 64

      subroutine test_pmxvf32gernn_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32gernn(cq, vu10, vu11, 7, 2)
      end subroutine test_pmxvf32gernn_u1_def

!CHECK-LABEL: @test_pmxvf32gernn_u1_def_
! LLVMIR:         %[[VAL_164:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_165:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_166:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_167:.*]] = load <16 x i8>, ptr %[[VAL_165]], align 16
! LLVMIR:         %[[VAL_168:.*]] = load <16 x i8>, ptr %[[VAL_164]], align 16
! LLVMIR:         %[[VAL_169:.*]] = load <512 x i1>, ptr %[[VAL_166]], align 64
! LLVMIR:         %[[VAL_170:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32gernn(<512 x i1> %[[VAL_169]], <16 x i8> %[[VAL_167]], <16 x i8> %[[VAL_168]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_170]], ptr %[[VAL_166]], align 64

      subroutine test_pmxvf32gernn_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32gernn(cq, vu10, vu11, 7_2, 2_1)
      end subroutine test_pmxvf32gernn_u1_non_def

!CHECK-LABEL: @test_pmxvf32gernn_u1_non_def_
! LLVMIR:         %[[VAL_171:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_172:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_173:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_174:.*]] = load <16 x i8>, ptr %[[VAL_172]], align 16
! LLVMIR:         %[[VAL_175:.*]] = load <16 x i8>, ptr %[[VAL_171]], align 16
! LLVMIR:         %[[VAL_176:.*]] = load <512 x i1>, ptr %[[VAL_173]], align 64
! LLVMIR:         %[[VAL_177:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32gernn(<512 x i1> %[[VAL_176]], <16 x i8> %[[VAL_174]], <16 x i8> %[[VAL_175]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_177]], ptr %[[VAL_173]], align 64

      subroutine test_pmxvf32gernn_r4_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32gernn(cq, vr40, vr41, 7, 2)
      end subroutine test_pmxvf32gernn_r4_def

!CHECK-LABEL: @test_pmxvf32gernn_r4_def_
! LLVMIR:         %[[VAL_178:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_179:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_180:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_181:.*]] = load <4 x float>, ptr %[[VAL_179]], align 16
! LLVMIR:         %[[VAL_182:.*]] = load <4 x float>, ptr %[[VAL_178]], align 16
! LLVMIR:         %[[VAL_183:.*]] = load <512 x i1>, ptr %[[VAL_180]], align 64
! LLVMIR:         %[[VAL_184:.*]] = bitcast <4 x float> %[[VAL_181]] to <16 x i8>
! LLVMIR:         %[[VAL_185:.*]] = bitcast <4 x float> %[[VAL_182]] to <16 x i8>
! LLVMIR:         %[[VAL_186:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32gernn(<512 x i1> %[[VAL_183]], <16 x i8> %[[VAL_184]], <16 x i8> %[[VAL_185]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_186]], ptr %[[VAL_180]], align 64

      subroutine test_pmxvf32gernn_r4_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32gernn(cq, vr40, vr41, 7_2, 2_1)
      end subroutine test_pmxvf32gernn_r4_non_def

!CHECK-LABEL: @test_pmxvf32gernn_r4_non_def_
! LLVMIR:         %[[VAL_187:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_188:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_189:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_190:.*]] = load <4 x float>, ptr %[[VAL_188]], align 16
! LLVMIR:         %[[VAL_191:.*]] = load <4 x float>, ptr %[[VAL_187]], align 16
! LLVMIR:         %[[VAL_192:.*]] = load <512 x i1>, ptr %[[VAL_189]], align 64
! LLVMIR:         %[[VAL_193:.*]] = bitcast <4 x float> %[[VAL_190]] to <16 x i8>
! LLVMIR:         %[[VAL_194:.*]] = bitcast <4 x float> %[[VAL_191]] to <16 x i8>
! LLVMIR:         %[[VAL_195:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32gernn(<512 x i1> %[[VAL_192]], <16 x i8> %[[VAL_193]], <16 x i8> %[[VAL_194]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_195]], ptr %[[VAL_189]], align 64

      subroutine test_pmxvf32gernp_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32gernp(cq, vu10, vu11, 7, 2)
      end subroutine test_pmxvf32gernp_u1_def

!CHECK-LABEL: @test_pmxvf32gernp_u1_def_
! LLVMIR:         %[[VAL_196:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_197:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_198:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_199:.*]] = load <16 x i8>, ptr %[[VAL_197]], align 16
! LLVMIR:         %[[VAL_200:.*]] = load <16 x i8>, ptr %[[VAL_196]], align 16
! LLVMIR:         %[[VAL_201:.*]] = load <512 x i1>, ptr %[[VAL_198]], align 64
! LLVMIR:         %[[VAL_202:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32gernp(<512 x i1> %[[VAL_201]], <16 x i8> %[[VAL_199]], <16 x i8> %[[VAL_200]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_202]], ptr %[[VAL_198]], align 64

      subroutine test_pmxvf32gernp_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32gernp(cq, vu10, vu11, 7_2, 2_1)
      end subroutine test_pmxvf32gernp_u1_non_def

!CHECK-LABEL: @test_pmxvf32gernp_u1_non_def_
! LLVMIR:         %[[VAL_203:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_204:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_205:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_206:.*]] = load <16 x i8>, ptr %[[VAL_204]], align 16
! LLVMIR:         %[[VAL_207:.*]] = load <16 x i8>, ptr %[[VAL_203]], align 16
! LLVMIR:         %[[VAL_208:.*]] = load <512 x i1>, ptr %[[VAL_205]], align 64
! LLVMIR:         %[[VAL_209:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32gernp(<512 x i1> %[[VAL_208]], <16 x i8> %[[VAL_206]], <16 x i8> %[[VAL_207]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_209]], ptr %[[VAL_205]], align 64

      subroutine test_pmxvf32gernp_r4_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32gernp(cq, vr40, vr41, 7, 2)
      end subroutine test_pmxvf32gernp_r4_def

!CHECK-LABEL: @test_pmxvf32gernp_r4_def_
! LLVMIR:         %[[VAL_210:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_211:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_212:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_213:.*]] = load <4 x float>, ptr %[[VAL_211]], align 16
! LLVMIR:         %[[VAL_214:.*]] = load <4 x float>, ptr %[[VAL_210]], align 16
! LLVMIR:         %[[VAL_215:.*]] = load <512 x i1>, ptr %[[VAL_212]], align 64
! LLVMIR:         %[[VAL_216:.*]] = bitcast <4 x float> %[[VAL_213]] to <16 x i8>
! LLVMIR:         %[[VAL_217:.*]] = bitcast <4 x float> %[[VAL_214]] to <16 x i8>
! LLVMIR:         %[[VAL_218:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32gernp(<512 x i1> %[[VAL_215]], <16 x i8> %[[VAL_216]], <16 x i8> %[[VAL_217]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_218]], ptr %[[VAL_212]], align 64

      subroutine test_pmxvf32gernp_r4_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32gernp(cq, vr40, vr41, 7_2, 2_1)
      end subroutine test_pmxvf32gernp_r4_non_def

!CHECK-LABEL: @test_pmxvf32gernp_r4_non_def_
! LLVMIR:         %[[VAL_219:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_220:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_221:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_222:.*]] = load <4 x float>, ptr %[[VAL_220]], align 16
! LLVMIR:         %[[VAL_223:.*]] = load <4 x float>, ptr %[[VAL_219]], align 16
! LLVMIR:         %[[VAL_224:.*]] = load <512 x i1>, ptr %[[VAL_221]], align 64
! LLVMIR:         %[[VAL_225:.*]] = bitcast <4 x float> %[[VAL_222]] to <16 x i8>
! LLVMIR:         %[[VAL_226:.*]] = bitcast <4 x float> %[[VAL_223]] to <16 x i8>
! LLVMIR:         %[[VAL_227:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32gernp(<512 x i1> %[[VAL_224]], <16 x i8> %[[VAL_225]], <16 x i8> %[[VAL_226]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_227]], ptr %[[VAL_221]], align 64

      subroutine test_pmxvf32gerpn_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32gerpn(cq, vu10, vu11, 7, 2)
      end subroutine test_pmxvf32gerpn_u1_def

!CHECK-LABEL: @test_pmxvf32gerpn_u1_def_
! LLVMIR:         %[[VAL_228:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_229:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_230:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_231:.*]] = load <16 x i8>, ptr %[[VAL_229]], align 16
! LLVMIR:         %[[VAL_232:.*]] = load <16 x i8>, ptr %[[VAL_228]], align 16
! LLVMIR:         %[[VAL_233:.*]] = load <512 x i1>, ptr %[[VAL_230]], align 64
! LLVMIR:         %[[VAL_234:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32gerpn(<512 x i1> %[[VAL_233]], <16 x i8> %[[VAL_231]], <16 x i8> %[[VAL_232]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_234]], ptr %[[VAL_230]], align 64

      subroutine test_pmxvf32gerpn_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32gerpn(cq, vu10, vu11, 7_2, 2_1)
      end subroutine test_pmxvf32gerpn_u1_non_def

!CHECK-LABEL: @test_pmxvf32gerpn_u1_non_def_
! LLVMIR:         %[[VAL_235:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_236:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_237:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_238:.*]] = load <16 x i8>, ptr %[[VAL_236]], align 16
! LLVMIR:         %[[VAL_239:.*]] = load <16 x i8>, ptr %[[VAL_235]], align 16
! LLVMIR:         %[[VAL_240:.*]] = load <512 x i1>, ptr %[[VAL_237]], align 64
! LLVMIR:         %[[VAL_241:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32gerpn(<512 x i1> %[[VAL_240]], <16 x i8> %[[VAL_238]], <16 x i8> %[[VAL_239]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_241]], ptr %[[VAL_237]], align 64

      subroutine test_pmxvf32gerpn_r4_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32gerpn(cq, vr40, vr41, 7, 2)
      end subroutine test_pmxvf32gerpn_r4_def

!CHECK-LABEL: @test_pmxvf32gerpn_r4_def_
! LLVMIR:         %[[VAL_242:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_243:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_244:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_245:.*]] = load <4 x float>, ptr %[[VAL_243]], align 16
! LLVMIR:         %[[VAL_246:.*]] = load <4 x float>, ptr %[[VAL_242]], align 16
! LLVMIR:         %[[VAL_247:.*]] = load <512 x i1>, ptr %[[VAL_244]], align 64
! LLVMIR:         %[[VAL_248:.*]] = bitcast <4 x float> %[[VAL_245]] to <16 x i8>
! LLVMIR:         %[[VAL_249:.*]] = bitcast <4 x float> %[[VAL_246]] to <16 x i8>
! LLVMIR:         %[[VAL_250:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32gerpn(<512 x i1> %[[VAL_247]], <16 x i8> %[[VAL_248]], <16 x i8> %[[VAL_249]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_250]], ptr %[[VAL_244]], align 64

      subroutine test_pmxvf32gerpn_r4_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32gerpn(cq, vr40, vr41, 7_2, 2_1)
      end subroutine test_pmxvf32gerpn_r4_non_def

!CHECK-LABEL: @test_pmxvf32gerpn_r4_non_def_
! LLVMIR:         %[[VAL_251:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_252:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_253:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_254:.*]] = load <4 x float>, ptr %[[VAL_252]], align 16
! LLVMIR:         %[[VAL_255:.*]] = load <4 x float>, ptr %[[VAL_251]], align 16
! LLVMIR:         %[[VAL_256:.*]] = load <512 x i1>, ptr %[[VAL_253]], align 64
! LLVMIR:         %[[VAL_257:.*]] = bitcast <4 x float> %[[VAL_254]] to <16 x i8>
! LLVMIR:         %[[VAL_258:.*]] = bitcast <4 x float> %[[VAL_255]] to <16 x i8>
! LLVMIR:         %[[VAL_259:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32gerpn(<512 x i1> %[[VAL_256]], <16 x i8> %[[VAL_257]], <16 x i8> %[[VAL_258]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_259]], ptr %[[VAL_253]], align 64

      subroutine test_pmxvf32gerpp_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32gerpp(cq, vu10, vu11, 7, 2)
      end subroutine test_pmxvf32gerpp_u1_def

!CHECK-LABEL: @test_pmxvf32gerpp_u1_def_
! LLVMIR:         %[[VAL_260:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_261:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_262:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_263:.*]] = load <16 x i8>, ptr %[[VAL_261]], align 16
! LLVMIR:         %[[VAL_264:.*]] = load <16 x i8>, ptr %[[VAL_260]], align 16
! LLVMIR:         %[[VAL_265:.*]] = load <512 x i1>, ptr %[[VAL_262]], align 64
! LLVMIR:         %[[VAL_266:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32gerpp(<512 x i1> %[[VAL_265]], <16 x i8> %[[VAL_263]], <16 x i8> %[[VAL_264]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_266]], ptr %[[VAL_262]], align 64

      subroutine test_pmxvf32gerpp_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvf32gerpp(cq, vu10, vu11, 7_2, 2_1)
      end subroutine test_pmxvf32gerpp_u1_non_def

!CHECK-LABEL: @test_pmxvf32gerpp_u1_non_def_
! LLVMIR:         %[[VAL_267:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_268:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_269:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_270:.*]] = load <16 x i8>, ptr %[[VAL_268]], align 16
! LLVMIR:         %[[VAL_271:.*]] = load <16 x i8>, ptr %[[VAL_267]], align 16
! LLVMIR:         %[[VAL_272:.*]] = load <512 x i1>, ptr %[[VAL_269]], align 64
! LLVMIR:         %[[VAL_273:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32gerpp(<512 x i1> %[[VAL_272]], <16 x i8> %[[VAL_270]], <16 x i8> %[[VAL_271]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_273]], ptr %[[VAL_269]], align 64

      subroutine test_pmxvf32gerpp_r4_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32gerpp(cq, vr40, vr41, 7, 2)
      end subroutine test_pmxvf32gerpp_r4_def

!CHECK-LABEL: @test_pmxvf32gerpp_r4_def_
! LLVMIR:         %[[VAL_274:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_275:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_276:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_277:.*]] = load <4 x float>, ptr %[[VAL_275]], align 16
! LLVMIR:         %[[VAL_278:.*]] = load <4 x float>, ptr %[[VAL_274]], align 16
! LLVMIR:         %[[VAL_279:.*]] = load <512 x i1>, ptr %[[VAL_276]], align 64
! LLVMIR:         %[[VAL_280:.*]] = bitcast <4 x float> %[[VAL_277]] to <16 x i8>
! LLVMIR:         %[[VAL_281:.*]] = bitcast <4 x float> %[[VAL_278]] to <16 x i8>
! LLVMIR:         %[[VAL_282:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32gerpp(<512 x i1> %[[VAL_279]], <16 x i8> %[[VAL_280]], <16 x i8> %[[VAL_281]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_282]], ptr %[[VAL_276]], align 64

      subroutine test_pmxvf32gerpp_r4_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(4)) vr40, vr41
      __vector_quad :: cq
      call mma_pmxvf32gerpp(cq, vr40, vr41, 7_2, 2_1)
      end subroutine test_pmxvf32gerpp_r4_non_def

!CHECK-LABEL: @test_pmxvf32gerpp_r4_non_def_
! LLVMIR:         %[[VAL_283:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_284:.*]] = alloca <4 x float>, i64 1, align 16
! LLVMIR:         %[[VAL_285:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_286:.*]] = load <4 x float>, ptr %[[VAL_284]], align 16
! LLVMIR:         %[[VAL_287:.*]] = load <4 x float>, ptr %[[VAL_283]], align 16
! LLVMIR:         %[[VAL_288:.*]] = load <512 x i1>, ptr %[[VAL_285]], align 64
! LLVMIR:         %[[VAL_289:.*]] = bitcast <4 x float> %[[VAL_286]] to <16 x i8>
! LLVMIR:         %[[VAL_290:.*]] = bitcast <4 x float> %[[VAL_287]] to <16 x i8>
! LLVMIR:         %[[VAL_291:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf32gerpp(<512 x i1> %[[VAL_288]], <16 x i8> %[[VAL_289]], <16 x i8> %[[VAL_290]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_291]], ptr %[[VAL_285]], align 64

      subroutine test_pmxvf64ger_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64ger(cq, cp, vu10, 7, 2)
      end subroutine test_pmxvf64ger_u1_def

!CHECK-LABEL: @test_pmxvf64ger_u1_def_
! LLVMIR:         %[[VAL_292:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_293:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_294:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_295:.*]] = load <256 x i1>, ptr %[[VAL_294]], align 32
! LLVMIR:         %[[VAL_296:.*]] = load <16 x i8>, ptr %[[VAL_292]], align 16
! LLVMIR:         %[[VAL_297:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64ger(<256 x i1> %[[VAL_295]], <16 x i8> %[[VAL_296]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_297]], ptr %[[VAL_293]], align 64

      subroutine test_pmxvf64ger_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64ger(cq, cp, vu10, 7_2, 2_1)
      end subroutine test_pmxvf64ger_u1_non_def

!CHECK-LABEL: @test_pmxvf64ger_u1_non_def_
! LLVMIR:         %[[VAL_298:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_299:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_300:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_301:.*]] = load <256 x i1>, ptr %[[VAL_300]], align 32
! LLVMIR:         %[[VAL_302:.*]] = load <16 x i8>, ptr %[[VAL_298]], align 16
! LLVMIR:         %[[VAL_303:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64ger(<256 x i1> %[[VAL_301]], <16 x i8> %[[VAL_302]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_303]], ptr %[[VAL_299]], align 64

      subroutine test_pmxvf64ger_r8_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64ger(cq, cp, vr80, 7, 2)
      end subroutine test_pmxvf64ger_r8_def

!CHECK-LABEL: @test_pmxvf64ger_r8_def_
! LLVMIR:         %[[VAL_304:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_305:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_306:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_307:.*]] = load <256 x i1>, ptr %[[VAL_306]], align 32
! LLVMIR:         %[[VAL_308:.*]] = load <2 x double>, ptr %[[VAL_304]], align 16
! LLVMIR:         %[[VAL_309:.*]] = bitcast <2 x double> %[[VAL_308]] to <16 x i8>
! LLVMIR:         %[[VAL_310:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64ger(<256 x i1> %[[VAL_307]], <16 x i8> %[[VAL_309]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_310]], ptr %[[VAL_305]], align 64

      subroutine test_pmxvf64ger_r8_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64ger(cq, cp, vr80, 7_2, 2_1)
      end subroutine test_pmxvf64ger_r8_non_def

!CHECK-LABEL: @test_pmxvf64ger_r8_non_def_
! LLVMIR:         %[[VAL_311:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_312:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_313:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_314:.*]] = load <256 x i1>, ptr %[[VAL_313]], align 32
! LLVMIR:         %[[VAL_315:.*]] = load <2 x double>, ptr %[[VAL_311]], align 16
! LLVMIR:         %[[VAL_316:.*]] = bitcast <2 x double> %[[VAL_315]] to <16 x i8>
! LLVMIR:         %[[VAL_317:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64ger(<256 x i1> %[[VAL_314]], <16 x i8> %[[VAL_316]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_317]], ptr %[[VAL_312]], align 64

      subroutine test_pmxvf64gernn_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gernn(cq, cp, vu10, 7, 2)
      end subroutine test_pmxvf64gernn_u1_def

!CHECK-LABEL: @test_pmxvf64gernn_u1_def_
! LLVMIR:         %[[VAL_318:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_319:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_320:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_321:.*]] = load <256 x i1>, ptr %[[VAL_320]], align 32
! LLVMIR:         %[[VAL_322:.*]] = load <16 x i8>, ptr %[[VAL_318]], align 16
! LLVMIR:         %[[VAL_323:.*]] = load <512 x i1>, ptr %[[VAL_319]], align 64
! LLVMIR:         %[[VAL_324:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64gernn(<512 x i1> %[[VAL_323]], <256 x i1> %[[VAL_321]], <16 x i8> %[[VAL_322]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_324]], ptr %[[VAL_319]], align 64

      subroutine test_pmxvf64gernn_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gernn(cq, cp, vu10, 7_2, 2_1)
      end subroutine test_pmxvf64gernn_u1_non_def

!CHECK-LABEL: @test_pmxvf64gernn_u1_non_def_
! LLVMIR:         %[[VAL_325:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_326:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_327:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_328:.*]] = load <256 x i1>, ptr %[[VAL_327]], align 32
! LLVMIR:         %[[VAL_329:.*]] = load <16 x i8>, ptr %[[VAL_325]], align 16
! LLVMIR:         %[[VAL_330:.*]] = load <512 x i1>, ptr %[[VAL_326]], align 64
! LLVMIR:         %[[VAL_331:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64gernn(<512 x i1> %[[VAL_330]], <256 x i1> %[[VAL_328]], <16 x i8> %[[VAL_329]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_331]], ptr %[[VAL_326]], align 64

      subroutine test_pmxvf64gernn_r8_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gernn(cq, cp, vr80, 7, 2)
      end subroutine test_pmxvf64gernn_r8_def

!CHECK-LABEL: @test_pmxvf64gernn_r8_def_
! LLVMIR:         %[[VAL_332:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_333:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_334:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_335:.*]] = load <256 x i1>, ptr %[[VAL_334]], align 32
! LLVMIR:         %[[VAL_336:.*]] = load <2 x double>, ptr %[[VAL_332]], align 16
! LLVMIR:         %[[VAL_337:.*]] = load <512 x i1>, ptr %[[VAL_333]], align 64
! LLVMIR:         %[[VAL_338:.*]] = bitcast <2 x double> %[[VAL_336]] to <16 x i8>
! LLVMIR:         %[[VAL_339:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64gernn(<512 x i1> %[[VAL_337]], <256 x i1> %[[VAL_335]], <16 x i8> %[[VAL_338]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_339]], ptr %[[VAL_333]], align 64

      subroutine test_pmxvf64gernn_r8_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gernn(cq, cp, vr80, 7_2, 2_1)
      end subroutine test_pmxvf64gernn_r8_non_def

!CHECK-LABEL: @test_pmxvf64gernn_r8_non_def_
! LLVMIR:         %[[VAL_340:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_341:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_342:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_343:.*]] = load <256 x i1>, ptr %[[VAL_342]], align 32
! LLVMIR:         %[[VAL_344:.*]] = load <2 x double>, ptr %[[VAL_340]], align 16
! LLVMIR:         %[[VAL_345:.*]] = load <512 x i1>, ptr %[[VAL_341]], align 64
! LLVMIR:         %[[VAL_346:.*]] = bitcast <2 x double> %[[VAL_344]] to <16 x i8>
! LLVMIR:         %[[VAL_347:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64gernn(<512 x i1> %[[VAL_345]], <256 x i1> %[[VAL_343]], <16 x i8> %[[VAL_346]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_347]], ptr %[[VAL_341]], align 64

      subroutine test_pmxvf64gernp_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gernp(cq, cp, vu10, 7, 2)
      end subroutine test_pmxvf64gernp_u1_def

!CHECK-LABEL: @test_pmxvf64gernp_u1_def_
! LLVMIR:         %[[VAL_348:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_349:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_350:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_351:.*]] = load <256 x i1>, ptr %[[VAL_350]], align 32
! LLVMIR:         %[[VAL_352:.*]] = load <16 x i8>, ptr %[[VAL_348]], align 16
! LLVMIR:         %[[VAL_353:.*]] = load <512 x i1>, ptr %[[VAL_349]], align 64
! LLVMIR:         %[[VAL_354:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64gernp(<512 x i1> %[[VAL_353]], <256 x i1> %[[VAL_351]], <16 x i8> %[[VAL_352]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_354]], ptr %[[VAL_349]], align 64

      subroutine test_pmxvf64gernp_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gernp(cq, cp, vu10, 7_2, 2_1)
      end subroutine test_pmxvf64gernp_u1_non_def

!CHECK-LABEL: @test_pmxvf64gernp_u1_non_def_
! LLVMIR:         %[[VAL_355:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_356:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_357:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_358:.*]] = load <256 x i1>, ptr %[[VAL_357]], align 32
! LLVMIR:         %[[VAL_359:.*]] = load <16 x i8>, ptr %[[VAL_355]], align 16
! LLVMIR:         %[[VAL_360:.*]] = load <512 x i1>, ptr %[[VAL_356]], align 64
! LLVMIR:         %[[VAL_361:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64gernp(<512 x i1> %[[VAL_360]], <256 x i1> %[[VAL_358]], <16 x i8> %[[VAL_359]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_361]], ptr %[[VAL_356]], align 64

      subroutine test_pmxvf64gernp_r8_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gernp(cq, cp, vr80, 7, 2)
      end subroutine test_pmxvf64gernp_r8_def

!CHECK-LABEL: @test_pmxvf64gernp_r8_def_
! LLVMIR:         %[[VAL_362:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_363:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_364:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_365:.*]] = load <256 x i1>, ptr %[[VAL_364]], align 32
! LLVMIR:         %[[VAL_366:.*]] = load <2 x double>, ptr %[[VAL_362]], align 16
! LLVMIR:         %[[VAL_367:.*]] = load <512 x i1>, ptr %[[VAL_363]], align 64
! LLVMIR:         %[[VAL_368:.*]] = bitcast <2 x double> %[[VAL_366]] to <16 x i8>
! LLVMIR:         %[[VAL_369:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64gernp(<512 x i1> %[[VAL_367]], <256 x i1> %[[VAL_365]], <16 x i8> %[[VAL_368]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_369]], ptr %[[VAL_363]], align 64

      subroutine test_pmxvf64gernp_r8_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gernp(cq, cp, vr80, 7_2, 2_1)
      end subroutine test_pmxvf64gernp_r8_non_def

!CHECK-LABEL: @test_pmxvf64gernp_r8_non_def_
! LLVMIR:         %[[VAL_370:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_371:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_372:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_373:.*]] = load <256 x i1>, ptr %[[VAL_372]], align 32
! LLVMIR:         %[[VAL_374:.*]] = load <2 x double>, ptr %[[VAL_370]], align 16
! LLVMIR:         %[[VAL_375:.*]] = load <512 x i1>, ptr %[[VAL_371]], align 64
! LLVMIR:         %[[VAL_376:.*]] = bitcast <2 x double> %[[VAL_374]] to <16 x i8>
! LLVMIR:         %[[VAL_377:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64gernp(<512 x i1> %[[VAL_375]], <256 x i1> %[[VAL_373]], <16 x i8> %[[VAL_376]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_377]], ptr %[[VAL_371]], align 64

      subroutine test_pmxvf64gerpn_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gerpn(cq, cp, vu10, 7, 2)
      end subroutine test_pmxvf64gerpn_u1_def

!CHECK-LABEL: @test_pmxvf64gerpn_u1_def_
! LLVMIR:         %[[VAL_378:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_379:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_380:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_381:.*]] = load <256 x i1>, ptr %[[VAL_380]], align 32
! LLVMIR:         %[[VAL_382:.*]] = load <16 x i8>, ptr %[[VAL_378]], align 16
! LLVMIR:         %[[VAL_383:.*]] = load <512 x i1>, ptr %[[VAL_379]], align 64
! LLVMIR:         %[[VAL_384:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64gerpn(<512 x i1> %[[VAL_383]], <256 x i1> %[[VAL_381]], <16 x i8> %[[VAL_382]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_384]], ptr %[[VAL_379]], align 64

      subroutine test_pmxvf64gerpn_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gerpn(cq, cp, vu10, 7_2, 2_1)
      end subroutine test_pmxvf64gerpn_u1_non_def

!CHECK-LABEL: @test_pmxvf64gerpn_u1_non_def_
! LLVMIR:         %[[VAL_385:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_386:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_387:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_388:.*]] = load <256 x i1>, ptr %[[VAL_387]], align 32
! LLVMIR:         %[[VAL_389:.*]] = load <16 x i8>, ptr %[[VAL_385]], align 16
! LLVMIR:         %[[VAL_390:.*]] = load <512 x i1>, ptr %[[VAL_386]], align 64
! LLVMIR:         %[[VAL_391:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64gerpn(<512 x i1> %[[VAL_390]], <256 x i1> %[[VAL_388]], <16 x i8> %[[VAL_389]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_391]], ptr %[[VAL_386]], align 64

      subroutine test_pmxvf64gerpn_r8_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gerpn(cq, cp, vr80, 7, 2)
      end subroutine test_pmxvf64gerpn_r8_def

!CHECK-LABEL: @test_pmxvf64gerpn_r8_def_
! LLVMIR:         %[[VAL_392:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_393:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_394:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_395:.*]] = load <256 x i1>, ptr %[[VAL_394]], align 32
! LLVMIR:         %[[VAL_396:.*]] = load <2 x double>, ptr %[[VAL_392]], align 16
! LLVMIR:         %[[VAL_397:.*]] = load <512 x i1>, ptr %[[VAL_393]], align 64
! LLVMIR:         %[[VAL_398:.*]] = bitcast <2 x double> %[[VAL_396]] to <16 x i8>
! LLVMIR:         %[[VAL_399:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64gerpn(<512 x i1> %[[VAL_397]], <256 x i1> %[[VAL_395]], <16 x i8> %[[VAL_398]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_399]], ptr %[[VAL_393]], align 64

      subroutine test_pmxvf64gerpn_r8_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gerpn(cq, cp, vr80, 7_2, 2_1)
      end subroutine test_pmxvf64gerpn_r8_non_def

!CHECK-LABEL: @test_pmxvf64gerpn_r8_non_def_
! LLVMIR:         %[[VAL_400:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_401:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_402:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_403:.*]] = load <256 x i1>, ptr %[[VAL_402]], align 32
! LLVMIR:         %[[VAL_404:.*]] = load <2 x double>, ptr %[[VAL_400]], align 16
! LLVMIR:         %[[VAL_405:.*]] = load <512 x i1>, ptr %[[VAL_401]], align 64
! LLVMIR:         %[[VAL_406:.*]] = bitcast <2 x double> %[[VAL_404]] to <16 x i8>
! LLVMIR:         %[[VAL_407:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64gerpn(<512 x i1> %[[VAL_405]], <256 x i1> %[[VAL_403]], <16 x i8> %[[VAL_406]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_407]], ptr %[[VAL_401]], align 64

      subroutine test_pmxvf64gerpp_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gerpp(cq, cp, vu10, 7, 2)
      end subroutine test_pmxvf64gerpp_u1_def

!CHECK-LABEL: @test_pmxvf64gerpp_u1_def_
! LLVMIR:         %[[VAL_408:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_409:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_410:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_411:.*]] = load <256 x i1>, ptr %[[VAL_410]], align 32
! LLVMIR:         %[[VAL_412:.*]] = load <16 x i8>, ptr %[[VAL_408]], align 16
! LLVMIR:         %[[VAL_413:.*]] = load <512 x i1>, ptr %[[VAL_409]], align 64
! LLVMIR:         %[[VAL_414:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64gerpp(<512 x i1> %[[VAL_413]], <256 x i1> %[[VAL_411]], <16 x i8> %[[VAL_412]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_414]], ptr %[[VAL_409]], align 64

      subroutine test_pmxvf64gerpp_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gerpp(cq, cp, vu10, 7_2, 2_1)
      end subroutine test_pmxvf64gerpp_u1_non_def

!CHECK-LABEL: @test_pmxvf64gerpp_u1_non_def_
! LLVMIR:         %[[VAL_415:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_416:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_417:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_418:.*]] = load <256 x i1>, ptr %[[VAL_417]], align 32
! LLVMIR:         %[[VAL_419:.*]] = load <16 x i8>, ptr %[[VAL_415]], align 16
! LLVMIR:         %[[VAL_420:.*]] = load <512 x i1>, ptr %[[VAL_416]], align 64
! LLVMIR:         %[[VAL_421:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64gerpp(<512 x i1> %[[VAL_420]], <256 x i1> %[[VAL_418]], <16 x i8> %[[VAL_419]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_421]], ptr %[[VAL_416]], align 64

      subroutine test_pmxvf64gerpp_r8_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gerpp(cq, cp, vr80, 7, 2)
      end subroutine test_pmxvf64gerpp_r8_def

!CHECK-LABEL: @test_pmxvf64gerpp_r8_def_
! LLVMIR:         %[[VAL_422:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_423:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_424:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_425:.*]] = load <256 x i1>, ptr %[[VAL_424]], align 32
! LLVMIR:         %[[VAL_426:.*]] = load <2 x double>, ptr %[[VAL_422]], align 16
! LLVMIR:         %[[VAL_427:.*]] = load <512 x i1>, ptr %[[VAL_423]], align 64
! LLVMIR:         %[[VAL_428:.*]] = bitcast <2 x double> %[[VAL_426]] to <16 x i8>
! LLVMIR:         %[[VAL_429:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64gerpp(<512 x i1> %[[VAL_427]], <256 x i1> %[[VAL_425]], <16 x i8> %[[VAL_428]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_429]], ptr %[[VAL_423]], align 64

      subroutine test_pmxvf64gerpp_r8_non_def()
      use, intrinsic :: mma
      implicit none
      vector(real(8)) vr80
      __vector_pair :: cp
      __vector_quad :: cq
      call mma_pmxvf64gerpp(cq, cp, vr80, 7_2, 2_1)
      end subroutine test_pmxvf64gerpp_r8_non_def

!CHECK-LABEL: @test_pmxvf64gerpp_r8_non_def_
! LLVMIR:         %[[VAL_430:.*]] = alloca <2 x double>, i64 1, align 16
! LLVMIR:         %[[VAL_431:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_432:.*]] = alloca <256 x i1>, i64 1, align 32
! LLVMIR:         %[[VAL_433:.*]] = load <256 x i1>, ptr %[[VAL_432]], align 32
! LLVMIR:         %[[VAL_434:.*]] = load <2 x double>, ptr %[[VAL_430]], align 16
! LLVMIR:         %[[VAL_435:.*]] = load <512 x i1>, ptr %[[VAL_431]], align 64
! LLVMIR:         %[[VAL_436:.*]] = bitcast <2 x double> %[[VAL_434]] to <16 x i8>
! LLVMIR:         %[[VAL_437:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvf64gerpp(<512 x i1> %[[VAL_435]], <256 x i1> %[[VAL_433]], <16 x i8> %[[VAL_436]], i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_437]], ptr %[[VAL_431]], align 64

      subroutine test_pmxvi16ger2_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi16ger2(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi16ger2_u1_def

!CHECK-LABEL: @test_pmxvi16ger2_u1_def_
! LLVMIR:         %[[VAL_438:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_439:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_440:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_441:.*]] = load <16 x i8>, ptr %[[VAL_439]], align 16
! LLVMIR:         %[[VAL_442:.*]] = load <16 x i8>, ptr %[[VAL_438]], align 16
! LLVMIR:         %[[VAL_443:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2(<16 x i8> %[[VAL_441]], <16 x i8> %[[VAL_442]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_443]], ptr %[[VAL_440]], align 64

      subroutine test_pmxvi16ger2_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi16ger2(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi16ger2_u1_non_def

!CHECK-LABEL: @test_pmxvi16ger2_u1_non_def_
! LLVMIR:         %[[VAL_444:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_445:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_446:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_447:.*]] = load <16 x i8>, ptr %[[VAL_445]], align 16
! LLVMIR:         %[[VAL_448:.*]] = load <16 x i8>, ptr %[[VAL_444]], align 16
! LLVMIR:         %[[VAL_449:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2(<16 x i8> %[[VAL_447]], <16 x i8> %[[VAL_448]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_449]], ptr %[[VAL_446]], align 64

      subroutine test_pmxvi16ger2_i2_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_pmxvi16ger2(cq, vi20, vi21, 7, 7, 2)
      end subroutine test_pmxvi16ger2_i2_def

!CHECK-LABEL: @test_pmxvi16ger2_i2_def_
! LLVMIR:         %[[VAL_450:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_451:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_452:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_453:.*]] = load <8 x i16>, ptr %[[VAL_451]], align 16
! LLVMIR:         %[[VAL_454:.*]] = load <8 x i16>, ptr %[[VAL_450]], align 16
! LLVMIR:         %[[VAL_455:.*]] = bitcast <8 x i16> %[[VAL_453]] to <16 x i8>
! LLVMIR:         %[[VAL_456:.*]] = bitcast <8 x i16> %[[VAL_454]] to <16 x i8>
! LLVMIR:         %[[VAL_457:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2(<16 x i8> %[[VAL_455]], <16 x i8> %[[VAL_456]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_457]], ptr %[[VAL_452]], align 64

      subroutine test_pmxvi16ger2_i2_non_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_pmxvi16ger2(cq, vi20, vi21, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi16ger2_i2_non_def

!CHECK-LABEL: @test_pmxvi16ger2_i2_non_def_
! LLVMIR:         %[[VAL_458:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_459:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_460:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_461:.*]] = load <8 x i16>, ptr %[[VAL_459]], align 16
! LLVMIR:         %[[VAL_462:.*]] = load <8 x i16>, ptr %[[VAL_458]], align 16
! LLVMIR:         %[[VAL_463:.*]] = bitcast <8 x i16> %[[VAL_461]] to <16 x i8>
! LLVMIR:         %[[VAL_464:.*]] = bitcast <8 x i16> %[[VAL_462]] to <16 x i8>
! LLVMIR:         %[[VAL_465:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2(<16 x i8> %[[VAL_463]], <16 x i8> %[[VAL_464]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_465]], ptr %[[VAL_460]], align 64

      subroutine test_pmxvi16ger2pp_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi16ger2pp(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi16ger2pp_u1_def

!CHECK-LABEL: @test_pmxvi16ger2pp_u1_def_
! LLVMIR:         %[[VAL_466:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_467:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_468:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_469:.*]] = load <16 x i8>, ptr %[[VAL_467]], align 16
! LLVMIR:         %[[VAL_470:.*]] = load <16 x i8>, ptr %[[VAL_466]], align 16
! LLVMIR:         %[[VAL_471:.*]] = load <512 x i1>, ptr %[[VAL_468]], align 64
! LLVMIR:         %[[VAL_472:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2pp(<512 x i1> %[[VAL_471]], <16 x i8> %[[VAL_469]], <16 x i8> %[[VAL_470]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_472]], ptr %[[VAL_468]], align 64

      subroutine test_pmxvi16ger2pp_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi16ger2pp(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi16ger2pp_u1_non_def

!CHECK-LABEL: @test_pmxvi16ger2pp_u1_non_def_
! LLVMIR:         %[[VAL_473:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_474:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_475:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_476:.*]] = load <16 x i8>, ptr %[[VAL_474]], align 16
! LLVMIR:         %[[VAL_477:.*]] = load <16 x i8>, ptr %[[VAL_473]], align 16
! LLVMIR:         %[[VAL_478:.*]] = load <512 x i1>, ptr %[[VAL_475]], align 64
! LLVMIR:         %[[VAL_479:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2pp(<512 x i1> %[[VAL_478]], <16 x i8> %[[VAL_476]], <16 x i8> %[[VAL_477]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_479]], ptr %[[VAL_475]], align 64

      subroutine test_pmxvi16ger2pp_i2_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_pmxvi16ger2pp(cq, vi20, vi21, 7, 7, 2)
      end subroutine test_pmxvi16ger2pp_i2_def

!CHECK-LABEL: @test_pmxvi16ger2pp_i2_def_
! LLVMIR:         %[[VAL_480:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_481:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_482:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_483:.*]] = load <8 x i16>, ptr %[[VAL_481]], align 16
! LLVMIR:         %[[VAL_484:.*]] = load <8 x i16>, ptr %[[VAL_480]], align 16
! LLVMIR:         %[[VAL_485:.*]] = load <512 x i1>, ptr %[[VAL_482]], align 64
! LLVMIR:         %[[VAL_486:.*]] = bitcast <8 x i16> %[[VAL_483]] to <16 x i8>
! LLVMIR:         %[[VAL_487:.*]] = bitcast <8 x i16> %[[VAL_484]] to <16 x i8>
! LLVMIR:         %[[VAL_488:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2pp(<512 x i1> %[[VAL_485]], <16 x i8> %[[VAL_486]], <16 x i8> %[[VAL_487]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_488]], ptr %[[VAL_482]], align 64

      subroutine test_pmxvi16ger2pp_i2_non_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_pmxvi16ger2pp(cq, vi20, vi21, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi16ger2pp_i2_non_def

!CHECK-LABEL: @test_pmxvi16ger2pp_i2_non_def_
! LLVMIR:         %[[VAL_489:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_490:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_491:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_492:.*]] = load <8 x i16>, ptr %[[VAL_490]], align 16
! LLVMIR:         %[[VAL_493:.*]] = load <8 x i16>, ptr %[[VAL_489]], align 16
! LLVMIR:         %[[VAL_494:.*]] = load <512 x i1>, ptr %[[VAL_491]], align 64
! LLVMIR:         %[[VAL_495:.*]] = bitcast <8 x i16> %[[VAL_492]] to <16 x i8>
! LLVMIR:         %[[VAL_496:.*]] = bitcast <8 x i16> %[[VAL_493]] to <16 x i8>
! LLVMIR:         %[[VAL_497:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2pp(<512 x i1> %[[VAL_494]], <16 x i8> %[[VAL_495]], <16 x i8> %[[VAL_496]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_497]], ptr %[[VAL_491]], align 64

      subroutine test_pmxvi16ger2s_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi16ger2s(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi16ger2s_u1_def

!CHECK-LABEL: @test_pmxvi16ger2s_u1_def_
! LLVMIR:         %[[VAL_498:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_499:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_500:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_501:.*]] = load <16 x i8>, ptr %[[VAL_499]], align 16
! LLVMIR:         %[[VAL_502:.*]] = load <16 x i8>, ptr %[[VAL_498]], align 16
! LLVMIR:         %[[VAL_503:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2s(<16 x i8> %[[VAL_501]], <16 x i8> %[[VAL_502]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_503]], ptr %[[VAL_500]], align 64

      subroutine test_pmxvi16ger2s_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi16ger2s(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi16ger2s_u1_non_def

!CHECK-LABEL: @test_pmxvi16ger2s_u1_non_def_
! LLVMIR:         %[[VAL_504:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_505:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_506:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_507:.*]] = load <16 x i8>, ptr %[[VAL_505]], align 16
! LLVMIR:         %[[VAL_508:.*]] = load <16 x i8>, ptr %[[VAL_504]], align 16
! LLVMIR:         %[[VAL_509:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2s(<16 x i8> %[[VAL_507]], <16 x i8> %[[VAL_508]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_509]], ptr %[[VAL_506]], align 64

      subroutine test_pmxvi16ger2s_i2_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_pmxvi16ger2s(cq, vi20, vi21, 7, 7, 2)
      end subroutine test_pmxvi16ger2s_i2_def

!CHECK-LABEL: @test_pmxvi16ger2s_i2_def_
! LLVMIR:         %[[VAL_510:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_511:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_512:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_513:.*]] = load <8 x i16>, ptr %[[VAL_511]], align 16
! LLVMIR:         %[[VAL_514:.*]] = load <8 x i16>, ptr %[[VAL_510]], align 16
! LLVMIR:         %[[VAL_515:.*]] = bitcast <8 x i16> %[[VAL_513]] to <16 x i8>
! LLVMIR:         %[[VAL_516:.*]] = bitcast <8 x i16> %[[VAL_514]] to <16 x i8>
! LLVMIR:         %[[VAL_517:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2s(<16 x i8> %[[VAL_515]], <16 x i8> %[[VAL_516]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_517]], ptr %[[VAL_512]], align 64

      subroutine test_pmxvi16ger2s_i2_non_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_pmxvi16ger2s(cq, vi20, vi21, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi16ger2s_i2_non_def

!CHECK-LABEL: @test_pmxvi16ger2s_i2_non_def_
! LLVMIR:         %[[VAL_518:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_519:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_520:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_521:.*]] = load <8 x i16>, ptr %[[VAL_519]], align 16
! LLVMIR:         %[[VAL_522:.*]] = load <8 x i16>, ptr %[[VAL_518]], align 16
! LLVMIR:         %[[VAL_523:.*]] = bitcast <8 x i16> %[[VAL_521]] to <16 x i8>
! LLVMIR:         %[[VAL_524:.*]] = bitcast <8 x i16> %[[VAL_522]] to <16 x i8>
! LLVMIR:         %[[VAL_525:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2s(<16 x i8> %[[VAL_523]], <16 x i8> %[[VAL_524]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_525]], ptr %[[VAL_520]], align 64

      subroutine test_pmxvi16ger2spp_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi16ger2spp(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi16ger2spp_u1_def

!CHECK-LABEL: @test_pmxvi16ger2spp_u1_def_
! LLVMIR:         %[[VAL_526:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_527:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_528:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_529:.*]] = load <16 x i8>, ptr %[[VAL_527]], align 16
! LLVMIR:         %[[VAL_530:.*]] = load <16 x i8>, ptr %[[VAL_526]], align 16
! LLVMIR:         %[[VAL_531:.*]] = load <512 x i1>, ptr %[[VAL_528]], align 64
! LLVMIR:         %[[VAL_532:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2spp(<512 x i1> %[[VAL_531]], <16 x i8> %[[VAL_529]], <16 x i8> %[[VAL_530]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_532]], ptr %[[VAL_528]], align 64

      subroutine test_pmxvi16ger2spp_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi16ger2spp(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi16ger2spp_u1_non_def

!CHECK-LABEL: @test_pmxvi16ger2spp_u1_non_def_
! LLVMIR:         %[[VAL_533:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_534:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_535:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_536:.*]] = load <16 x i8>, ptr %[[VAL_534]], align 16
! LLVMIR:         %[[VAL_537:.*]] = load <16 x i8>, ptr %[[VAL_533]], align 16
! LLVMIR:         %[[VAL_538:.*]] = load <512 x i1>, ptr %[[VAL_535]], align 64
! LLVMIR:         %[[VAL_539:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2spp(<512 x i1> %[[VAL_538]], <16 x i8> %[[VAL_536]], <16 x i8> %[[VAL_537]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_539]], ptr %[[VAL_535]], align 64

      subroutine test_pmxvi16ger2spp_i2_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_pmxvi16ger2spp(cq, vi20, vi21, 7, 7, 2)
      end subroutine test_pmxvi16ger2spp_i2_def

!CHECK-LABEL: @test_pmxvi16ger2spp_i2_def_
! LLVMIR:         %[[VAL_540:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_541:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_542:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_543:.*]] = load <8 x i16>, ptr %[[VAL_541]], align 16
! LLVMIR:         %[[VAL_544:.*]] = load <8 x i16>, ptr %[[VAL_540]], align 16
! LLVMIR:         %[[VAL_545:.*]] = load <512 x i1>, ptr %[[VAL_542]], align 64
! LLVMIR:         %[[VAL_546:.*]] = bitcast <8 x i16> %[[VAL_543]] to <16 x i8>
! LLVMIR:         %[[VAL_547:.*]] = bitcast <8 x i16> %[[VAL_544]] to <16 x i8>
! LLVMIR:         %[[VAL_548:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2spp(<512 x i1> %[[VAL_545]], <16 x i8> %[[VAL_546]], <16 x i8> %[[VAL_547]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_548]], ptr %[[VAL_542]], align 64

      subroutine test_pmxvi16ger2spp_i2_non_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(2)) vi20, vi21
      __vector_quad :: cq
      call mma_pmxvi16ger2spp(cq, vi20, vi21, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi16ger2spp_i2_non_def

!CHECK-LABEL: @test_pmxvi16ger2spp_i2_non_def_
! LLVMIR:         %[[VAL_549:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_550:.*]] = alloca <8 x i16>, i64 1, align 16
! LLVMIR:         %[[VAL_551:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_552:.*]] = load <8 x i16>, ptr %[[VAL_550]], align 16
! LLVMIR:         %[[VAL_553:.*]] = load <8 x i16>, ptr %[[VAL_549]], align 16
! LLVMIR:         %[[VAL_554:.*]] = load <512 x i1>, ptr %[[VAL_551]], align 64
! LLVMIR:         %[[VAL_555:.*]] = bitcast <8 x i16> %[[VAL_552]] to <16 x i8>
! LLVMIR:         %[[VAL_556:.*]] = bitcast <8 x i16> %[[VAL_553]] to <16 x i8>
! LLVMIR:         %[[VAL_557:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi16ger2spp(<512 x i1> %[[VAL_554]], <16 x i8> %[[VAL_555]], <16 x i8> %[[VAL_556]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_557]], ptr %[[VAL_551]], align 64


      subroutine test_pmxvi4ger8_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi4ger8(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi4ger8_def

!CHECK-LABEL: @test_pmxvi4ger8_def_
! LLVMIR:         %[[VAL_558:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_559:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_560:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_561:.*]] = load <16 x i8>, ptr %[[VAL_559]], align 16
! LLVMIR:         %[[VAL_562:.*]] = load <16 x i8>, ptr %[[VAL_558]], align 16
! LLVMIR:         %[[VAL_563:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi4ger8(<16 x i8> %[[VAL_561]], <16 x i8> %[[VAL_562]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_563]], ptr %[[VAL_560]], align 64

      subroutine test_pmxvi4ger8_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi4ger8(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi4ger8_non_def

!CHECK-LABEL: @test_pmxvi4ger8_non_def_
! LLVMIR:         %[[VAL_564:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_565:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_566:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_567:.*]] = load <16 x i8>, ptr %[[VAL_565]], align 16
! LLVMIR:         %[[VAL_568:.*]] = load <16 x i8>, ptr %[[VAL_564]], align 16
! LLVMIR:         %[[VAL_569:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi4ger8(<16 x i8> %[[VAL_567]], <16 x i8> %[[VAL_568]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_569]], ptr %[[VAL_566]], align 64

      subroutine test_pmxvi4ger8pp_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi4ger8pp(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi4ger8pp_def

!CHECK-LABEL: @test_pmxvi4ger8pp_def_
! LLVMIR:         %[[VAL_570:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_571:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_572:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_573:.*]] = load <16 x i8>, ptr %[[VAL_571]], align 16
! LLVMIR:         %[[VAL_574:.*]] = load <16 x i8>, ptr %[[VAL_570]], align 16
! LLVMIR:         %[[VAL_575:.*]] = load <512 x i1>, ptr %[[VAL_572]], align 64
! LLVMIR:         %[[VAL_576:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi4ger8pp(<512 x i1> %[[VAL_575]], <16 x i8> %[[VAL_573]], <16 x i8> %[[VAL_574]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_576]], ptr %[[VAL_572]], align 64

      subroutine test_pmxvi4ger8pp_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi4ger8pp(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi4ger8pp_non_def

!CHECK-LABEL: @test_pmxvi4ger8pp_non_def_
! LLVMIR:         %[[VAL_577:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_578:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_579:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_580:.*]] = load <16 x i8>, ptr %[[VAL_578]], align 16
! LLVMIR:         %[[VAL_581:.*]] = load <16 x i8>, ptr %[[VAL_577]], align 16
! LLVMIR:         %[[VAL_582:.*]] = load <512 x i1>, ptr %[[VAL_579]], align 64
! LLVMIR:         %[[VAL_583:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi4ger8pp(<512 x i1> %[[VAL_582]], <16 x i8> %[[VAL_580]], <16 x i8> %[[VAL_581]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_583]], ptr %[[VAL_579]], align 64

      subroutine test_pmxvi8ger4_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi8ger4(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi8ger4_u1_def

!CHECK-LABEL: @test_pmxvi8ger4_u1_def_
! LLVMIR:         %[[VAL_584:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_585:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_586:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_587:.*]] = load <16 x i8>, ptr %[[VAL_585]], align 16
! LLVMIR:         %[[VAL_588:.*]] = load <16 x i8>, ptr %[[VAL_584]], align 16
! LLVMIR:         %[[VAL_589:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4(<16 x i8> %[[VAL_587]], <16 x i8> %[[VAL_588]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_589]], ptr %[[VAL_586]], align 64

      subroutine test_pmxvi8ger4_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi8ger4(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi8ger4_u1_non_def

!CHECK-LABEL: @test_pmxvi8ger4_u1_non_def_
! LLVMIR:         %[[VAL_590:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_591:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_592:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_593:.*]] = load <16 x i8>, ptr %[[VAL_591]], align 16
! LLVMIR:         %[[VAL_594:.*]] = load <16 x i8>, ptr %[[VAL_590]], align 16
! LLVMIR:         %[[VAL_595:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4(<16 x i8> %[[VAL_593]], <16 x i8> %[[VAL_594]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_595]], ptr %[[VAL_592]], align 64

      subroutine test_pmxvi8ger4_i1_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_pmxvi8ger4(cq, vi10, vi11, 7, 7, 2)
      end subroutine test_pmxvi8ger4_i1_def

!CHECK-LABEL: @test_pmxvi8ger4_i1_def_
! LLVMIR:         %[[VAL_596:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_597:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_598:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_599:.*]] = load <16 x i8>, ptr %[[VAL_597]], align 16
! LLVMIR:         %[[VAL_600:.*]] = load <16 x i8>, ptr %[[VAL_596]], align 16
! LLVMIR:         %[[VAL_601:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4(<16 x i8> %[[VAL_599]], <16 x i8> %[[VAL_600]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_601]], ptr %[[VAL_598]], align 64

      subroutine test_pmxvi8ger4_i1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_pmxvi8ger4(cq, vi10, vi11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi8ger4_i1_non_def

!CHECK-LABEL: @test_pmxvi8ger4_i1_non_def_
! LLVMIR:         %[[VAL_602:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_603:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_604:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_605:.*]] = load <16 x i8>, ptr %[[VAL_603]], align 16
! LLVMIR:         %[[VAL_606:.*]] = load <16 x i8>, ptr %[[VAL_602]], align 16
! LLVMIR:         %[[VAL_607:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4(<16 x i8> %[[VAL_605]], <16 x i8> %[[VAL_606]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_607]], ptr %[[VAL_604]], align 64

      subroutine test_pmxvi8ger4pp_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi8ger4pp(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi8ger4pp_u1_def

!CHECK-LABEL: @test_pmxvi8ger4pp_u1_def_
! LLVMIR:         %[[VAL_608:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_609:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_610:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_611:.*]] = load <16 x i8>, ptr %[[VAL_609]], align 16
! LLVMIR:         %[[VAL_612:.*]] = load <16 x i8>, ptr %[[VAL_608]], align 16
! LLVMIR:         %[[VAL_613:.*]] = load <512 x i1>, ptr %[[VAL_610]], align 64
! LLVMIR:         %[[VAL_614:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4pp(<512 x i1> %[[VAL_613]], <16 x i8> %[[VAL_611]], <16 x i8> %[[VAL_612]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_614]], ptr %[[VAL_610]], align 64

      subroutine test_pmxvi8ger4pp_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi8ger4pp(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi8ger4pp_u1_non_def

!CHECK-LABEL: @test_pmxvi8ger4pp_u1_non_def_
! LLVMIR:         %[[VAL_615:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_616:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_617:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_618:.*]] = load <16 x i8>, ptr %[[VAL_616]], align 16
! LLVMIR:         %[[VAL_619:.*]] = load <16 x i8>, ptr %[[VAL_615]], align 16
! LLVMIR:         %[[VAL_620:.*]] = load <512 x i1>, ptr %[[VAL_617]], align 64
! LLVMIR:         %[[VAL_621:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4pp(<512 x i1> %[[VAL_620]], <16 x i8> %[[VAL_618]], <16 x i8> %[[VAL_619]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_621]], ptr %[[VAL_617]], align 64

      subroutine test_pmxvi8ger4pp_i1_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_pmxvi8ger4pp(cq, vi10, vi11, 7, 7, 2)
      end subroutine test_pmxvi8ger4pp_i1_def

!CHECK-LABEL: @test_pmxvi8ger4pp_i1_def_
! LLVMIR:         %[[VAL_622:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_623:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_624:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_625:.*]] = load <16 x i8>, ptr %[[VAL_623]], align 16
! LLVMIR:         %[[VAL_626:.*]] = load <16 x i8>, ptr %[[VAL_622]], align 16
! LLVMIR:         %[[VAL_627:.*]] = load <512 x i1>, ptr %[[VAL_624]], align 64
! LLVMIR:         %[[VAL_628:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4pp(<512 x i1> %[[VAL_627]], <16 x i8> %[[VAL_625]], <16 x i8> %[[VAL_626]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_628]], ptr %[[VAL_624]], align 64

      subroutine test_pmxvi8ger4pp_i1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_pmxvi8ger4pp(cq, vi10, vi11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi8ger4pp_i1_non_def

!CHECK-LABEL: @test_pmxvi8ger4pp_i1_non_def_
! LLVMIR:         %[[VAL_629:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_630:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_631:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_632:.*]] = load <16 x i8>, ptr %[[VAL_630]], align 16
! LLVMIR:         %[[VAL_633:.*]] = load <16 x i8>, ptr %[[VAL_629]], align 16
! LLVMIR:         %[[VAL_634:.*]] = load <512 x i1>, ptr %[[VAL_631]], align 64
! LLVMIR:         %[[VAL_635:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4pp(<512 x i1> %[[VAL_634]], <16 x i8> %[[VAL_632]], <16 x i8> %[[VAL_633]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_635]], ptr %[[VAL_631]], align 64

      subroutine test_pmxvi8ger4spp_u1_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi8ger4spp(cq, vu10, vu11, 7, 7, 2)
      end subroutine test_pmxvi8ger4spp_u1_def

!CHECK-LABEL: @test_pmxvi8ger4spp_u1_def_
! LLVMIR:         %[[VAL_636:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_637:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_638:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_639:.*]] = load <16 x i8>, ptr %[[VAL_637]], align 16
! LLVMIR:         %[[VAL_640:.*]] = load <16 x i8>, ptr %[[VAL_636]], align 16
! LLVMIR:         %[[VAL_641:.*]] = load <512 x i1>, ptr %[[VAL_638]], align 64
! LLVMIR:         %[[VAL_642:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4spp(<512 x i1> %[[VAL_641]], <16 x i8> %[[VAL_639]], <16 x i8> %[[VAL_640]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_642]], ptr %[[VAL_638]], align 64

      subroutine test_pmxvi8ger4spp_u1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(unsigned(1)) vu10, vu11
      __vector_quad :: cq
      call mma_pmxvi8ger4spp(cq, vu10, vu11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi8ger4spp_u1_non_def

!CHECK-LABEL: @test_pmxvi8ger4spp_u1_non_def_
! LLVMIR:         %[[VAL_643:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_644:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_645:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_646:.*]] = load <16 x i8>, ptr %[[VAL_644]], align 16
! LLVMIR:         %[[VAL_647:.*]] = load <16 x i8>, ptr %[[VAL_643]], align 16
! LLVMIR:         %[[VAL_648:.*]] = load <512 x i1>, ptr %[[VAL_645]], align 64
! LLVMIR:         %[[VAL_649:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4spp(<512 x i1> %[[VAL_648]], <16 x i8> %[[VAL_646]], <16 x i8> %[[VAL_647]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_649]], ptr %[[VAL_645]], align 64

      subroutine test_pmxvi8ger4spp_i1_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_pmxvi8ger4spp(cq, vi10, vi11, 7, 7, 2)
      end subroutine test_pmxvi8ger4spp_i1_def

!CHECK-LABEL: @test_pmxvi8ger4spp_i1_def_
! LLVMIR:         %[[VAL_650:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_651:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_652:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_653:.*]] = load <16 x i8>, ptr %[[VAL_651]], align 16
! LLVMIR:         %[[VAL_654:.*]] = load <16 x i8>, ptr %[[VAL_650]], align 16
! LLVMIR:         %[[VAL_655:.*]] = load <512 x i1>, ptr %[[VAL_652]], align 64
! LLVMIR:         %[[VAL_656:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4spp(<512 x i1> %[[VAL_655]], <16 x i8> %[[VAL_653]], <16 x i8> %[[VAL_654]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_656]], ptr %[[VAL_652]], align 64

      subroutine test_pmxvi8ger4spp_i1_non_def()
      use, intrinsic :: mma
      implicit none
      vector(integer(1)) vi10, vi11
      __vector_quad :: cq
      call mma_pmxvi8ger4spp(cq, vi10, vi11, 7_2, 7_1, 2_8)
      end subroutine test_pmxvi8ger4spp_i1_non_def

!CHECK-LABEL: @test_pmxvi8ger4spp_i1_non_def_
! LLVMIR:         %[[VAL_657:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_658:.*]] = alloca <16 x i8>, i64 1, align 16
! LLVMIR:         %[[VAL_659:.*]] = alloca <512 x i1>, i64 1, align 64
! LLVMIR:         %[[VAL_660:.*]] = load <16 x i8>, ptr %[[VAL_658]], align 16
! LLVMIR:         %[[VAL_661:.*]] = load <16 x i8>, ptr %[[VAL_657]], align 16
! LLVMIR:         %[[VAL_662:.*]] = load <512 x i1>, ptr %[[VAL_659]], align 64
! LLVMIR:         %[[VAL_663:.*]] = call <512 x i1> @llvm.ppc.mma.pmxvi8ger4spp(<512 x i1> %[[VAL_662]], <16 x i8> %[[VAL_660]], <16 x i8> %[[VAL_661]], i32 7, i32 7, i32 2)
! LLVMIR:         store <512 x i1> %[[VAL_663]], ptr %[[VAL_659]], align 64
