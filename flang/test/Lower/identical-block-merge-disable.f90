! Test disable identical block merge in the canonicalizer pass in bbc.
! Temporary fix for issue #1021.
! RUN: bbc -hlfir=false %s -o - | FileCheck %s

MODULE DMUMPS_SOL_LR
IMPLICIT NONE

TYPE BLR_STRUC_T
  INTEGER, DIMENSION(:), POINTER  :: PANELS_L 
  INTEGER, DIMENSION(:), POINTER  :: PANELS_U
  INTEGER, DIMENSION(:), POINTER :: BEGS_BLR_STATIC
END TYPE BLR_STRUC_T

TYPE(BLR_STRUC_T), POINTER, DIMENSION(:), SAVE :: BLR_ARRAY

CONTAINS

SUBROUTINE DMUMPS_SOL_FWD_LR_SU( IWHDLR, MTYPE )

  INTEGER, INTENT(IN) :: IWHDLR, MTYPE
  INTEGER :: NPARTSASS, NB_BLR

  IF (MTYPE.EQ.1) THEN
    IF ( associated( BLR_ARRAY(IWHDLR)%PANELS_L ) ) THEN
      NPARTSASS = size( BLR_ARRAY(IWHDLR)%PANELS_L )
      NB_BLR = size( BLR_ARRAY(IWHDLR)%BEGS_BLR_STATIC ) - 1
    ENDIF
  ELSE
    IF ( associated( BLR_ARRAY(IWHDLR)%PANELS_U ) ) THEN
      NPARTSASS = size( BLR_ARRAY(IWHDLR)%PANELS_U )
      NB_BLR = size( BLR_ARRAY(IWHDLR)%BEGS_BLR_STATIC ) - 1
    ENDIF
  ENDIF

END SUBROUTINE DMUMPS_SOL_FWD_LR_SU 

END MODULE DMUMPS_SOL_LR

! CHECK-LABEL: func @_QMdmumps_sol_lrPdmumps_sol_fwd_lr_su(
! CHECK-SAME:                                              %[[VAL_0:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<i32>{{.*}}) {
! CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 0 : i64
! CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = fir.address_of(@_QMdmumps_sol_lrEblr_array) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>
! CHECK:         %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "nb_blr", uniq_name = "_QMdmumps_sol_lrFdmumps_sol_fwd_lr_suEnb_blr"}
! CHECK:         %[[VAL_7:.*]] = fir.alloca i32 {bindc_name = "npartsass", uniq_name = "_QMdmumps_sol_lrFdmumps_sol_fwd_lr_suEnpartsass"}
! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:         %[[VAL_9:.*]] = arith.cmpi eq, %[[VAL_8]], %[[VAL_4]] : i32
! CHECK:         cond_br %[[VAL_9]], ^bb1, ^bb3
! CHECK:       ^bb1:
! CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>
! CHECK:         %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_10]], %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> i64
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_11]]#0 : (index) -> i64
! CHECK:         %[[VAL_15:.*]] = arith.subi %[[VAL_13]], %[[VAL_14]] : i64
! CHECK:         %[[VAL_16:.*]] = fir.coordinate_of %[[VAL_10]], %[[VAL_15]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, i64) -> !fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
! CHECK:         %[[VAL_17:.*]] = fir.field_index panels_l, !fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>
! CHECK:         %[[VAL_18:.*]] = fir.coordinate_of %[[VAL_16]], %[[VAL_17]] : (!fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_19:.*]] = fir.load %[[VAL_18]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_20:.*]] = fir.box_addr %[[VAL_19]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
! CHECK:         %[[VAL_22:.*]] = arith.cmpi ne, %[[VAL_21]], %[[VAL_2]] : i64
! CHECK:         cond_br %[[VAL_22]], ^bb2, ^bb5
! CHECK:       ^bb2:
! CHECK:         %[[VAL_23:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>
! CHECK:         %[[VAL_24:.*]]:3 = fir.box_dims %[[VAL_23]], %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_25:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i32) -> i64
! CHECK:         %[[VAL_27:.*]] = fir.convert %[[VAL_24]]#0 : (index) -> i64
! CHECK:         %[[VAL_28:.*]] = arith.subi %[[VAL_26]], %[[VAL_27]] : i64
! CHECK:         %[[VAL_29:.*]] = fir.coordinate_of %[[VAL_23]], %[[VAL_28]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, i64) -> !fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
! CHECK:         %[[VAL_30:.*]] = fir.coordinate_of %[[VAL_29]], %[[VAL_17]] : (!fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_31:.*]] = fir.load %[[VAL_30]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_32:.*]]:3 = fir.box_dims %[[VAL_31]], %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_33:.*]] = fir.convert %[[VAL_32]]#1 : (index) -> i32
! CHECK:         fir.store %[[VAL_33]] to %[[VAL_7]] : !fir.ref<i32>
! CHECK:         %[[VAL_34:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>
! CHECK:         %[[VAL_35:.*]]:3 = fir.box_dims %[[VAL_34]], %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_36:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i32) -> i64
! CHECK:         %[[VAL_38:.*]] = fir.convert %[[VAL_35]]#0 : (index) -> i64
! CHECK:         %[[VAL_39:.*]] = arith.subi %[[VAL_37]], %[[VAL_38]] : i64
! CHECK:         %[[VAL_40:.*]] = fir.coordinate_of %[[VAL_34]], %[[VAL_39]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, i64) -> !fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
! CHECK:         %[[VAL_41:.*]] = fir.field_index begs_blr_static, !fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>
! CHECK:         %[[VAL_42:.*]] = fir.coordinate_of %[[VAL_40]], %[[VAL_41]] : (!fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_43:.*]] = fir.load %[[VAL_42]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_44:.*]]:3 = fir.box_dims %[[VAL_43]], %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_45:.*]] = fir.convert %[[VAL_44]]#1 : (index) -> i32
! CHECK:         %[[VAL_46:.*]] = arith.subi %[[VAL_45]], %[[VAL_4]] : i32
! CHECK:         fir.store %[[VAL_46]] to %[[VAL_6]] : !fir.ref<i32>
! CHECK:         br ^bb5
! CHECK:       ^bb3:
! CHECK:         %[[VAL_47:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>
! CHECK:         %[[VAL_48:.*]]:3 = fir.box_dims %[[VAL_47]], %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_49:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_50:.*]] = fir.convert %[[VAL_49]] : (i32) -> i64
! CHECK:         %[[VAL_51:.*]] = fir.convert %[[VAL_48]]#0 : (index) -> i64
! CHECK:         %[[VAL_52:.*]] = arith.subi %[[VAL_50]], %[[VAL_51]] : i64
! CHECK:         %[[VAL_53:.*]] = fir.coordinate_of %[[VAL_47]], %[[VAL_52]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, i64) -> !fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
! CHECK:         %[[VAL_54:.*]] = fir.field_index panels_u, !fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>
! CHECK:         %[[VAL_55:.*]] = fir.coordinate_of %[[VAL_53]], %[[VAL_54]] : (!fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_56:.*]] = fir.load %[[VAL_55]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_57:.*]] = fir.box_addr %[[VAL_56]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:         %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
! CHECK:         %[[VAL_59:.*]] = arith.cmpi ne, %[[VAL_58]], %[[VAL_2]] : i64
! CHECK:         cond_br %[[VAL_59]], ^bb4, ^bb5
! CHECK:       ^bb4:
! CHECK:         %[[VAL_60:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>
! CHECK:         %[[VAL_61:.*]]:3 = fir.box_dims %[[VAL_60]], %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_62:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_63:.*]] = fir.convert %[[VAL_62]] : (i32) -> i64
! CHECK:         %[[VAL_64:.*]] = fir.convert %[[VAL_61]]#0 : (index) -> i64
! CHECK:         %[[VAL_65:.*]] = arith.subi %[[VAL_63]], %[[VAL_64]] : i64
! CHECK:         %[[VAL_66:.*]] = fir.coordinate_of %[[VAL_60]], %[[VAL_65]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, i64) -> !fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
! CHECK:         %[[VAL_67:.*]] = fir.coordinate_of %[[VAL_66]], %[[VAL_54]] : (!fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_68:.*]] = fir.load %[[VAL_67]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_69:.*]]:3 = fir.box_dims %[[VAL_68]], %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_70:.*]] = fir.convert %[[VAL_69]]#1 : (index) -> i32
! CHECK:         fir.store %[[VAL_70]] to %[[VAL_7]] : !fir.ref<i32>
! CHECK:         %[[VAL_71:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>
! CHECK:         %[[VAL_72:.*]]:3 = fir.box_dims %[[VAL_71]], %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_73:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_74:.*]] = fir.convert %[[VAL_73]] : (i32) -> i64
! CHECK:         %[[VAL_75:.*]] = fir.convert %[[VAL_72]]#0 : (index) -> i64
! CHECK:         %[[VAL_76:.*]] = arith.subi %[[VAL_74]], %[[VAL_75]] : i64
! CHECK:         %[[VAL_77:.*]] = fir.coordinate_of %[[VAL_71]], %[[VAL_76]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, i64) -> !fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>
! CHECK:         %[[VAL_78:.*]] = fir.field_index begs_blr_static, !fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>
! CHECK:         %[[VAL_79:.*]] = fir.coordinate_of %[[VAL_77]], %[[VAL_78]] : (!fir.ref<!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_80:.*]] = fir.load %[[VAL_79]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_81:.*]]:3 = fir.box_dims %[[VAL_80]], %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_82:.*]] = fir.convert %[[VAL_81]]#1 : (index) -> i32
! CHECK:         %[[VAL_83:.*]] = arith.subi %[[VAL_82]], %[[VAL_4]] : i32
! CHECK:         fir.store %[[VAL_83]] to %[[VAL_6]] : !fir.ref<i32>
! CHECK:         br ^bb5
! CHECK:       ^bb5:
! CHECK:         return
! CHECK:       }

