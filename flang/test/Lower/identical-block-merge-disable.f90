! Test disable identical block merge in the canonicalizer pass in bbc.
! Temporary fix for issue #1021.
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

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

! CHECK-LABEL: func.func @_QMdmumps_sol_lrPdmumps_sol_fwd_lr_su(
! CHECK-SAME:                                              %[[VAL_0:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<i32>{{.*}}) {
! CHECK-DAG:     %[[VAL_2:.*]] = fir.address_of(@_QMdmumps_sol_lrEblr_array) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>
! CHECK-DAG:     %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMdmumps_sol_lrEblr_array"}
! CHECK-DAG:     %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{.*}} arg 1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMdmumps_sol_lrFdmumps_sol_fwd_lr_suEiwhdlr"}
! CHECK-DAG:     %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{.*}} arg 2 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMdmumps_sol_lrFdmumps_sol_fwd_lr_suEmtype"}
! CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:         %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_6]], %{{.*}} : i32
! CHECK:         fir.if %[[VAL_7]] {
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMdmumps_sol_lrTblr_struc_t{panels_l:!fir.box<!fir.ptr<!fir.array<?xi32>>>,panels_u:!fir.box<!fir.ptr<!fir.array<?xi32>>>,begs_blr_static:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_10:.*]] = hlfir.designate %[[VAL_8]] (%{{.*}}) : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<{{.*}}>>>>, i64) -> !fir.ref<!fir.type<{{.*}}>>
! CHECK:           %[[VAL_11:.*]] = hlfir.designate %[[VAL_10]]{"panels_l"} {fortran_attrs = #fir.var_attrs<pointer>}
! CHECK:           %[[VAL_12:.*]] = fir.load %[[VAL_11]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_13:.*]] = fir.box_addr %[[VAL_12]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:           %[[VAL_14:.*]] = arith.cmpi ne, %{{.*}}, %{{.*}} : i64
! CHECK:           fir.if %[[VAL_14]] {
! CHECK:             {{.*}} = hlfir.designate %{{.*}}{"panels_l"}
! CHECK:             hlfir.assign %{{.*}} to %{{.*}}#0 : i32, !fir.ref<i32>
! CHECK:             {{.*}} = hlfir.designate %{{.*}}{"begs_blr_static"}
! CHECK:             hlfir.assign %{{.*}} to %{{.*}}#0 : i32, !fir.ref<i32>
! CHECK:           }
! CHECK:         } else {
! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_8_else:.*]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<{{.*}}>>>>>
! CHECK:           %[[VAL_16:.*]] = hlfir.designate %[[VAL_15]] (%{{.*}}) : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<{{.*}}>>>>, i64) -> !fir.ref<!fir.type<{{.*}}>>
! CHECK:           %[[VAL_17:.*]] = hlfir.designate %[[VAL_16]]{"panels_u"} {fortran_attrs = #fir.var_attrs<pointer>}
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_17]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_19:.*]] = fir.box_addr %[[VAL_18]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:           %[[VAL_20:.*]] = arith.cmpi ne, %{{.*}}, %{{.*}} : i64
! CHECK:           fir.if %[[VAL_20]] {
! CHECK:             {{.*}} = hlfir.designate %{{.*}}{"panels_u"}
! CHECK:             hlfir.assign %{{.*}} to %{{.*}}#0 : i32, !fir.ref<i32>
! CHECK:             {{.*}} = hlfir.designate %{{.*}}{"begs_blr_static"}
! CHECK:             hlfir.assign %{{.*}} to %{{.*}}#0 : i32, !fir.ref<i32>
! CHECK:           }
! CHECK:         }
! CHECK:         return
! CHECK:       }

