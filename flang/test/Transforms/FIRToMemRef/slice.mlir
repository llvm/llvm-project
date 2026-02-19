// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// converted based on example:
// subroutine copy(a,b)
//   integer, parameter :: n = 5, m = 7
//   integer, dimension(n,m) :: a, b, c
// 
//   a(2:4:2,1:7:3) = b(1:5:3,2:7:2)
// end
// 
// program main
//   integer, parameter :: n = 5, m = 7
//   integer, dimension(n,m) :: a, b, c
// 
//   do j = 1, m
//     do i = 1, n
//       a(i,j) = 0
//       b(i,j) = i + (j-1) * m
//     enddo
//   enddo
// 
//   call copy(a,b)
// 
//   print *, a
// end


// CHECK-LABEL: func.func @slice_2d
// CHECK:       %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[C3:.*]] = arith.constant 3 : index
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[C7:.*]] = arith.constant 7 : index
// CHECK:       %[[C5:.*]] = arith.constant 5 : index
// CHECK:       %[[UNDEF:.*]] = fir.undefined !fir.dscope
// CHECK:       %[[SHAPE:.*]] = fir.shape %[[C5]], %[[C7]] : (index, index) -> !fir.shape<2>
// CHECK:       %[[SLICE:.*]] = fir.slice %[[C1]], %[[C5]], %[[C3]], %[[C2]], %[[C7]], %[[C2]] : (index, index, index, index, index, index) -> !fir.slice<2>
// CHECK:       [[DECLARE:%[0-9]+]] = fir.declare %arg1(%[[SHAPE]]) dummy_scope %[[UNDEF]] {uniq_name = "b"} : (!fir.ref<!fir.array<5x7xi32>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<5x7xi32>>
// CHECK:       [[EMBOX:%[0-9]+]] = fir.embox [[DECLARE]](%[[SHAPE]]) [%[[SLICE]]] : (!fir.ref<!fir.array<5x7xi32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<2x3xi32>>
// CHECK:       %[[C1_0:.*]] = arith.constant 1 : index
// CHECK:       [[ADD1:%[0-9]+]] = arith.addi %[[C3]], %[[C1_0]] : index
// CHECK:       scf.for %[[ARG2:.*]] = %[[C1]] to [[ADD1]] step %[[C1]] {
// CHECK:         %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:         [[ADD2:%[0-9]+]] = arith.addi %[[C2]], %[[C1_1]] : index
// CHECK:         scf.for %[[ARG3:.*]] = %[[C1]] to [[ADD2]] step %[[C1]] {
// CHECK:           [[BOXADDR:%[0-9]+]] = fir.box_addr [[EMBOX]] : (!fir.box<!fir.array<2x3xi32>>) -> !fir.ref<!fir.array<2x3xi32>>
// CHECK:           [[CONVERT:%[0-9]+]] = fir.convert [[DECLARE]] : (!fir.ref<!fir.array<5x7xi32>>) -> memref<7x5xi32>
// CHECK:           %[[C1_2:.*]] = arith.constant 1 : index
// CHECK:           %[[C1_3:.*]] = arith.constant 1 : index
// CHECK:           [[SUB1:%[0-9]+]] = arith.subi %[[ARG3]], %[[C1_3]] : index
// CHECK:           [[MUL1:%[0-9]+]] = arith.muli [[SUB1]], %[[C3]] : index
// CHECK:           [[SUB2:%[0-9]+]] = arith.subi %[[C1]], %[[C1_2]] : index
// CHECK:           [[ADD3:%[0-9]+]] = arith.addi [[MUL1]], [[SUB2]] : index
// CHECK:           %[[C1_4:.*]] = arith.constant 1 : index
// CHECK:           [[SUB3:%[0-9]+]] = arith.subi %[[ARG2]], %[[C1_4]] : index
// CHECK:           [[MUL2:%[0-9]+]] = arith.muli [[SUB3]], %[[C2]] : index
// CHECK:           [[SUB4:%[0-9]+]] = arith.subi %[[C2]], %[[C1_2]] : index
// CHECK:           [[ADD4:%[0-9]+]] = arith.addi [[MUL2]], [[SUB4]] : index
// CHECK:           [[LOAD:%[0-9]+]] = memref.load [[CONVERT]][[[ADD4]], [[ADD3]]] : memref<7x5xi32>
func.func @slice_2d(%arg0: !fir.ref<!fir.array<5x7xi32>>, %arg1: !fir.ref<!fir.array<5x7xi32>>) {
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index
  %c5 = arith.constant 5 : index
  %0 = fir.undefined !fir.dscope
  %shape = fir.shape %c5, %c7 : (index, index) -> !fir.shape<2>
  %slice = fir.slice %c1, %c5, %c3, %c2, %c7, %c2 : (index, index, index, index, index, index) -> !fir.slice<2>
  %2 = fir.declare %arg1(%shape) dummy_scope %0 {uniq_name = "b"} : (!fir.ref<!fir.array<5x7xi32>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<5x7xi32>>
  %9 = fir.embox %2(%shape)[%slice] : (!fir.ref<!fir.array<5x7xi32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<2x3xi32>>
  %c1_0 = arith.constant 1 : index
  %11 = arith.addi %c3, %c1_0 : index
  scf.for %arg2 = %c1 to %11 step %c1 {
    %c1_1 = arith.constant 1 : index
    %12 = arith.addi %c2, %c1_1 : index
    scf.for %arg3 = %c1 to %12 step %c1 {
      %13 = fir.array_coor %9(%shape) %arg3, %arg2 : (!fir.box<!fir.array<2x3xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
      %14 = fir.load %13 : !fir.ref<i32>
    }
  }
  return
}

// subroutine copy(a,b)
//   integer, parameter :: n = 5, m = 7
//   integer, dimension(n, m, m) :: a, b, c
// 
//   a(2:4:2,1:7:3, 2:5:2) = b(1:5:3,2:7:2, 3:7:4)
// end
// 
// program main
//   integer, parameter :: n = 5, m = 7
//   integer, dimension(n, m, m) :: a, b, c
// 
//   do k = 1, m
//     do j = 1, m
//       do i = 1, n
//         a(i, j, k) = 0
//         b(i, j, k) = i + (j-1) * m + (k-1) * m * n
//       enddo
//     enddo
//   enddo
// 
//   call copy(a,b)
// 
//   print *, a
// end
// CHECK-LABEL: func.func @slice_3d
// CHECK:       %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[C3:.*]] = arith.constant 3 : index
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[C7:.*]] = arith.constant 7 : index
// CHECK:       %[[C5:.*]] = arith.constant 5 : index
// CHECK:       %[[DUMMY_SCOPE:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:       [[SHAPE:%[0-9]+]] = fir.shape %[[C5]], %[[C7]], %[[C7]] : (index, index, index) -> !fir.shape<3>
// CHECK:       [[DECLARE_A:%[0-9]+]] = fir.declare %arg0([[SHAPE]]) dummy_scope %[[DUMMY_SCOPE]] {uniq_name = "_QFcopyEa"} : (!fir.ref<!fir.array<5x7x7xi32>>, !fir.shape<3>, !fir.dscope) -> !fir.ref<!fir.array<5x7x7xi32>>
// CHECK:       [[DECLARE_B:%[0-9]+]] = fir.declare %arg1([[SHAPE]]) dummy_scope %[[DUMMY_SCOPE]] {uniq_name = "_QFcopyEb"} : (!fir.ref<!fir.array<5x7x7xi32>>, !fir.shape<3>, !fir.dscope) -> !fir.ref<!fir.array<5x7x7xi32>>
// CHECK:       [[ALLOCA:%.*]] = memref.alloca() {bindc_name = "c", uniq_name = "_QFcopyEc"} : memref<7x7x5xi32>
// CHECK:       [[REFC:%[0-9]+]] = fir.convert [[ALLOCA]] : (memref<7x7x5xi32>) -> !fir.ref<!fir.array<5x7x7xi32>>
// CHECK:       [[DECLARE_C:%[0-9]+]] = fir.declare [[REFC]]([[SHAPE]]) {uniq_name = "_QFcopyEc"} : (!fir.ref<!fir.array<5x7x7xi32>>, !fir.shape<3>) -> !fir.ref<!fir.array<5x7x7xi32>>
// CHECK:       [[ADDR:%[0-9]+]] = fir.address_of(@_QFcopyECm) : !fir.ref<i32>
// CHECK:       %[[SLICE:[0-9]+]] = fir.slice %[[C1]], %[[C5]], %[[C3]], %[[C2]], %[[C7]], %[[C2]], %[[C3]], %[[C7]], %[[C4]] : (index, index, index, index, index, index, index, index, index) -> !fir.slice<3>
// CHECK:       [[EMBOX:%[0-9]+]] = fir.embox [[DECLARE_B]]([[SHAPE]]) [%[[SLICE]]] : (!fir.ref<!fir.array<5x7x7xi32>>, !fir.shape<3>, !fir.slice<3>) -> !fir.box<!fir.array<2x3x2xi32>>
// CHECK:       fir.do_loop %[[ARG2:.*]] = %[[C1]] to %[[C2]] step %[[C1]] unordered {
// CHECK:         fir.do_loop %[[ARG3:.*]] = %[[C1]] to %[[C3]] step %[[C1]] unordered {
// CHECK:           fir.do_loop %[[ARG4:.*]] = %[[C1]] to %[[C2]] step %[[C1]] unordered {
// CHECK:             [[BOXADDR:%[0-9]+]] = fir.box_addr [[EMBOX]] : (!fir.box<!fir.array<2x3x2xi32>>) -> !fir.ref<!fir.array<2x3x2xi32>>
// CHECK:             [[CONVERT2:%[0-9]+]] = fir.convert [[DECLARE_B]] : (!fir.ref<!fir.array<5x7x7xi32>>) -> memref<7x7x5xi32>
// CHECK:             %[[C1_0:.*]] = arith.constant 1 : index
// CHECK:             %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:             [[SUB1:%[0-9]+]] = arith.subi %[[ARG4]], %[[C1_1]] : index
// CHECK:             [[MUL1:%[0-9]+]] = arith.muli [[SUB1]], %[[C3]] : index
// CHECK:             [[SUB2:%[0-9]+]] = arith.subi %[[C1]], %[[C1_0]] : index
// CHECK:             [[ADD1:%[0-9]+]] = arith.addi [[MUL1]], [[SUB2]] : index
// CHECK:             %[[C1_2:.*]] = arith.constant 1 : index
// CHECK:             [[SUB3:%[0-9]+]] = arith.subi %[[ARG3]], %[[C1_2]] : index
// CHECK:             [[MUL2:%[0-9]+]] = arith.muli [[SUB3]], %[[C2]] : index
// CHECK:             [[SUB4:%[0-9]+]] = arith.subi %[[C2]], %[[C1_0]] : index
// CHECK:             [[ADD2:%[0-9]+]] = arith.addi [[MUL2]], [[SUB4]] : index
// CHECK:             %[[C1_3:.*]] = arith.constant 1 : index
// CHECK:             [[SUB5:%[0-9]+]] = arith.subi %[[ARG2]], %[[C1_3]] : index
// CHECK:             [[MUL3:%[0-9]+]] = arith.muli [[SUB5]], %[[C4]] : index
// CHECK:             [[SUB6:%[0-9]+]] = arith.subi %[[C3]], %[[C1_0]] : index
// CHECK:             [[ADD3:%[0-9]+]] = arith.addi [[MUL3]], [[SUB6]] : index
// CHECK:             [[LOAD:%[0-9]+]] = memref.load [[CONVERT2]][[[ADD3]], [[ADD2]], [[ADD1]]] : memref<7x7x5xi32>
func.func @slice_3d(%arg0: !fir.ref<!fir.array<5x7x7xi32>> {fir.bindc_name = "a", llvm.nocapture}, %arg1: !fir.ref<!fir.array<5x7x7xi32>> {fir.bindc_name = "b", llvm.nocapture}) attributes {fir.internal_name = "_QPcopy"} {
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index
  %c5 = arith.constant 5 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c5, %c7, %c7 : (index, index, index) -> !fir.shape<3>
  %2 = fir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFcopyEa"} : (!fir.ref<!fir.array<5x7x7xi32>>, !fir.shape<3>, !fir.dscope) -> !fir.ref<!fir.array<5x7x7xi32>>
  %3 = fir.declare %arg1(%1) dummy_scope %0 {uniq_name = "_QFcopyEb"} : (!fir.ref<!fir.array<5x7x7xi32>>, !fir.shape<3>, !fir.dscope) -> !fir.ref<!fir.array<5x7x7xi32>>
  %4 = fir.alloca !fir.array<5x7x7xi32> {bindc_name = "c", uniq_name = "_QFcopyEc"}
  %5 = fir.declare %4(%1) {uniq_name = "_QFcopyEc"} : (!fir.ref<!fir.array<5x7x7xi32>>, !fir.shape<3>) -> !fir.ref<!fir.array<5x7x7xi32>>
  %6 = fir.address_of(@_QFcopyECm) : !fir.ref<i32>
  %10 = fir.slice %c1, %c5, %c3, %c2, %c7, %c2, %c3, %c7, %c4 : (index, index, index, index, index, index, index, index, index) -> !fir.slice<3>
  %11 = fir.embox %3(%1) [%10] : (!fir.ref<!fir.array<5x7x7xi32>>, !fir.shape<3>, !fir.slice<3>) -> !fir.box<!fir.array<2x3x2xi32>>
  fir.do_loop %arg2 = %c1 to %c2 step %c1 unordered {
    fir.do_loop %arg3 = %c1 to %c3 step %c1 unordered {
      fir.do_loop %arg4 = %c1 to %c2 step %c1 unordered {
        %14 = fir.array_coor %11 %arg4, %arg3, %arg2 : (!fir.box<!fir.array<2x3x2xi32>>, index, index, index) -> !fir.ref<i32>
        %15 = fir.load %14 : !fir.ref<i32>
      }
    }
  }
  return
}

// CHECK-LABEL: func.func @extract_row
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[C3:.*]] = arith.constant 3 : index
// CHECK:       %[[DUMMY_SCOPE:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:       [[SHAPE:%[0-9]+]] = fir.shape %[[C3]], %[[C3]] : (index, index) -> !fir.shape<2>
// CHECK:       [[DECLARE:%[0-9]+]] = fir.declare %arg0([[SHAPE]]) dummy_scope %[[DUMMY_SCOPE]] {uniq_name = "_QFextract_rowEmatrix"} : (!fir.ref<!fir.array<3x3xi32>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<3x3xi32>>
// CHECK:       %[[UNDEF:.*]] = fir.undefined index
// CHECK:       %[[SLICE:[0-9]+]] = fir.slice %[[C2]], %[[UNDEF]], %[[UNDEF]], %[[C1]], %[[C3]], %[[C1]] : (index, index, index, index, index, index) -> !fir.slice<2>
// CHECK:       [[EMBOX:%[0-9]+]] = fir.embox [[DECLARE]]([[SHAPE]]) [%[[SLICE]]] : (!fir.ref<!fir.array<3x3xi32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<3xi32>>
// CHECK:       %[[C1_0:.*]] = arith.constant 1 : index
// CHECK:       [[ADD1:%[0-9]+]] = arith.addi %[[C3]], %[[C1_0]] : index
// CHECK:       scf.for %[[ARG1:.*]] = %[[C1]] to [[ADD1]] step %[[C1]] {
// CHECK:         [[BOXADDR:%[0-9]+]] = fir.box_addr [[EMBOX]] : (!fir.box<!fir.array<3xi32>>) -> !fir.ref<!fir.array<3xi32>>
// CHECK:         [[CONVERT:%[0-9]+]] = fir.convert [[DECLARE]] : (!fir.ref<!fir.array<3x3xi32>>) -> memref<3x3xi32>
// CHECK:         %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:         [[SUB1:%[0-9]+]] = arith.subi %[[C2]], %[[C1_1]] : index
// CHECK:         %[[C1_2:.*]] = arith.constant 1 : index
// CHECK:         [[SUB2:%[0-9]+]] = arith.subi %[[ARG1]], %[[C1_2]] : index
// CHECK:         [[MUL1:%[0-9]+]] = arith.muli [[SUB2]], %[[C1]] : index
// CHECK:         [[SUB3:%[0-9]+]] = arith.subi %[[C1]], %[[C1_1]] : index
// CHECK:         [[ADD2:%[0-9]+]] = arith.addi [[MUL1]], [[SUB3]] : index
// CHECK:         [[LOAD:%[0-9]+]] = memref.load [[CONVERT]][[[ADD2]], [[SUB1]]] : memref<3x3xi32>
func.func @extract_row(%arg0: !fir.ref<!fir.array<3x3xi32>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c3, %c3 : (index, index) -> !fir.shape<2>
  %2 = fir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFextract_rowEmatrix"} : (!fir.ref<!fir.array<3x3xi32>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<3x3xi32>>
  %5 = fir.undefined index
  %6 = fir.slice %c2, %5, %5, %c1, %c3, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  %7 = fir.embox %2(%1) [%6] : (!fir.ref<!fir.array<3x3xi32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<3xi32>>
  %c1_0 = arith.constant 1 : index
  %8 = arith.addi %c3, %c1_0 : index
  scf.for %arg2 = %c1 to %8 step %c1 {
    %9 = fir.array_coor %7 %arg2 : (!fir.box<!fir.array<3xi32>>, index) -> !fir.ref<i32>
    %10 = fir.load %9 : !fir.ref<i32>
  }
  return
}


// CHECK-LABEL: func.func @extract_column
// CHECK:       %[[C10:.*]] = arith.constant 10 : index
// CHECK:       %[[C11:.*]] = arith.constant 11 : index
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:       %[[C5:.*]] = arith.constant 5 : index
// CHECK:       %[[C100:.*]] = arith.constant 100 : index
// CHECK:       %[[DUMMY_SCOPE:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:       %[[ADDR:.*]] = fir.address_of(@_QFextract_columnECn) : !fir.ref<i32>
// CHECK:       [[DECLARE_N:%[0-9]+]] = fir.declare %[[ADDR]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "n"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:       [[SHAPE:%[0-9]+]] = fir.shape %[[C100]], %[[C5]] : (index, index) -> !fir.shape<2>
// CHECK:       [[DECLARE_TMP:%[0-9]+]] = fir.declare %arg0([[SHAPE]]) dummy_scope %[[DUMMY_SCOPE]] {uniq_name = "tmp"} : (!fir.ref<!fir.array<100x5xf32>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<100x5xf32>>
// CHECK:       %[[UNDEF:.*]] = fir.undefined index
// CHECK:       %[[SLICE:[0-9]+]] = fir.slice %[[C1]], %[[C100]], %[[C11]], %[[C1]], %[[UNDEF]], %[[UNDEF]] : (index, index, index, index, index, index) -> !fir.slice<2>
// CHECK:       [[EMBOX:%[0-9]+]] = fir.embox [[DECLARE_TMP]]([[SHAPE]]) [%[[SLICE]]] : (!fir.ref<!fir.array<100x5xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<10xf32>>
// CHECK:       %[[C1_0:.*]] = arith.constant 1 : index
// CHECK:       [[ADD1:%[0-9]+]] = arith.addi %[[C10]], %[[C1_0]] : index
// CHECK:       scf.for %[[ARG1:.*]] = %[[C1]] to [[ADD1]] step %[[C1]] {
// CHECK:         [[BOXADDR:%[0-9]+]] = fir.box_addr [[EMBOX]] : (!fir.box<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>>
// CHECK:         [[CONVERT:%[0-9]+]] = fir.convert [[DECLARE_TMP]] : (!fir.ref<!fir.array<100x5xf32>>) -> memref<5x100xf32>
// CHECK:         %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:         [[SUB1:%[0-9]+]] = arith.subi %[[C1]], %[[C1_1]] : index
// CHECK:         %[[C1_2:.*]] = arith.constant 1 : index
// CHECK:         [[SUB2:%[0-9]+]] = arith.subi %[[ARG1]], %[[C1_2]] : index
// CHECK:         [[MUL1:%[0-9]+]] = arith.muli [[SUB2]], %[[C11]] : index
// CHECK:         [[SUB3:%[0-9]+]] = arith.subi %[[C1]], %[[C1_1]] : index
// CHECK:         [[ADD2:%[0-9]+]] = arith.addi [[MUL1]], [[SUB3]] : index
// CHECK:         memref.store %[[CST]], [[CONVERT]][[[SUB1]], [[ADD2]]] : memref<5x100xf32>
func.func @extract_column(%arg0: !fir.ref<!fir.array<100x5xf32>> {fir.bindc_name = "tmp", llvm.nocapture}) attributes {fir.internal_name = "_QPextract_column"} {
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : f32
  %c5 = arith.constant 5 : index
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QFextract_columnECn) : !fir.ref<i32>
  %2 = fir.declare %1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "n"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %5 = fir.shape %c100, %c5 : (index, index) -> !fir.shape<2>
  %6 = fir.declare %arg0(%5) dummy_scope %0 {uniq_name = "tmp"} : (!fir.ref<!fir.array<100x5xf32>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<100x5xf32>>
  %8 = fir.undefined index
  %9 = fir.slice %c1, %c100, %c11, %c1, %8, %8 : (index, index, index, index, index, index) -> !fir.slice<2>
  %10 = fir.embox %6(%5) [%9] : (!fir.ref<!fir.array<100x5xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<10xf32>>
  %c1_2 = arith.constant 1 : index
  %11 = arith.addi %c10, %c1_2 : index
  scf.for %arg1 = %c1 to %11 step %c1 {
    %12 = fir.array_coor %10 %arg1 : (!fir.box<!fir.array<10xf32>>, index) -> !fir.ref<f32>
    fir.store %cst to %12 : !fir.ref<f32>
  }
  return
}


// CHECK-LABEL: func.func @noslice
// CHECK:       %[[C7:.*]] = arith.constant 7 : index
// CHECK:       %[[C_NEG1:.*]] = arith.constant -1 : index
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       [[ALLOCA:%.*]] = fir.alloca !fir.box<!fir.heap<!fir.type<_QMcodaTcodaK6{samples:!fir.array<7xf32>}>>> {bindc_name = "c", uniq_name = "_QMcodaFtrythisEc"}
// CHECK:       [[DECLARE:%.*]] = fir.declare [[ALLOCA]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMcodaFtrythisEc"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMcodaTcodaK6{samples:!fir.array<7xf32>}>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<_QMcodaTcodaK6{samples:!fir.array<7xf32>}>>>>
// CHECK:       [[LOADBOX:%.*]] = fir.load [[DECLARE]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMcodaTcodaK6{samples:!fir.array<7xf32>}>>>>
// CHECK:       [[BOXADDR:%.*]] = fir.box_addr [[LOADBOX]] : (!fir.box<!fir.heap<!fir.type<_QMcodaTcodaK6{samples:!fir.array<7xf32>}>>>) -> !fir.heap<!fir.type<_QMcodaTcodaK6{samples:!fir.array<7xf32>}>>
// CHECK:       [[COORD:%.*]] = fir.coordinate_of [[BOXADDR]], samples : (!fir.heap<!fir.type<_QMcodaTcodaK6{samples:!fir.array<7xf32>}>>) -> !fir.ref<!fir.array<7xf32>>
// CHECK:       [[EMBOX:%.*]] = fir.embox [[COORD]] : (!fir.ref<!fir.array<7xf32>>) -> !fir.box<!fir.array<7xf32>>
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       fir.do_loop %[[ARG0:.*]] = %[[C1]] to %[[C7]] step %[[C1]] unordered {
// CHECK:         [[ADD1:%.*]] = arith.addi %[[ARG0]], %[[C_NEG1]] : index
// CHECK:         [[SHIFT:%.*]] = fir.shift %[[C0]] : (index) -> !fir.shift<1>
// CHECK:         [[BOXADDR2:%.*]] = fir.box_addr [[EMBOX]] : (!fir.box<!fir.array<7xf32>>) -> !fir.ref<!fir.array<7xf32>>
// CHECK:         [[CONVERT:%.*]] = fir.convert [[BOXADDR2]] : (!fir.ref<!fir.array<7xf32>>) -> memref<7xf32>
// CHECK:         %[[C1_0:.*]] = arith.constant 1 : index
// CHECK:         [[ADD2:%.*]] = arith.addi %[[ARG0]], %[[C_NEG1]] : index
// CHECK:         [[SUB1:%.*]] = arith.subi [[ADD2]], %[[C0]] : index
// CHECK:         [[MUL1:%.*]] = arith.muli [[SUB1]], %[[C1_0]] : index
// CHECK:         [[SUB2:%[0-9]+]] = arith.subi %[[C0]], %[[C0]] : index
// CHECK:         [[ADD3:%.*]] = arith.addi [[MUL1]], [[SUB2]] : index
// CHECK:         [[LOADVAL:%.*]] = memref.load [[CONVERT]][[[ADD3]]] : memref<7xf32>
func.func @noslice() {
  %c7 = arith.constant 7 : index
  %c-1 = arith.constant -1 : index
  %c0 = arith.constant 0 : index
  %7 = fir.alloca !fir.box<!fir.heap<!fir.type<_QMcodaTcodaK6{samples:!fir.array<7xf32>}>>> {bindc_name = "c", uniq_name = "_QMcodaFtrythisEc"}
  %10 = fir.declare %7 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMcodaFtrythisEc"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMcodaTcodaK6{samples:!fir.array<7xf32>}>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<_QMcodaTcodaK6{samples:!fir.array<7xf32>}>>>>
  %27 = fir.load %10 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMcodaTcodaK6{samples:!fir.array<7xf32>}>>>>
  %28 = fir.box_addr %27 : (!fir.box<!fir.heap<!fir.type<_QMcodaTcodaK6{samples:!fir.array<7xf32>}>>>) -> !fir.heap<!fir.type<_QMcodaTcodaK6{samples:!fir.array<7xf32>}>>
  %31 = fir.coordinate_of %28, samples : (!fir.heap<!fir.type<_QMcodaTcodaK6{samples:!fir.array<7xf32>}>>) -> !fir.ref<!fir.array<7xf32>>
  %33 = fir.embox %31 : (!fir.ref<!fir.array<7xf32>>) -> !fir.box<!fir.array<7xf32>>
  %c1_0 = arith.constant 1 : index
  fir.do_loop %arg0 = %c1_0 to %c7 step %c1_0 unordered {
    %56 = arith.addi %arg0, %c-1 : index
    %57 = fir.shift %c0 : (index) -> !fir.shift<1>
    %58 = fir.array_coor %33(%57) %56 : (!fir.box<!fir.array<7xf32>>, !fir.shift<1>, index) -> !fir.ref<f32>
    %60 = fir.load %58 : !fir.ref<f32>
  }
  return
}

// CHECK-LABEL: func.func @array_coor_slice() attributes {fir.bindc_name = "tf4a", noinline}
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C_NEG1:.*]] = arith.constant -1 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C_NEG6:.*]] = arith.constant -6 : index
// CHECK: %[[C5:.*]] = arith.constant 5 : index
// CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<i32>
// CHECK: %[[CONVERT1:.*]] = fir.convert %[[ALLOCA]] : (memref<i32>) -> !fir.ref<i32>
// CHECK: %[[DUMMY_SCOPE:[0-9]+]] = fir.dummy_scope : !fir.dscope
// CHECK: %[[ADDR:[0-9]+]] = fir.address_of(@_QFFsECindex) : !fir.ref<!fir.array<5xi32>>
// CHECK: %[[SHAPE:[0-9]+]] = fir.shape %[[C5]] : (index) -> !fir.shape<1>
// CHECK: %[[DECLARE:[0-9]+]] = fir.declare %[[ADDR]](%[[SHAPE]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFFsECindex"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<5xi32>>
// CHECK: %[[LOAD:[0-9]+]] = memref.load %[[ALLOCA]][] : memref<i32>
// CHECK: %[[INDEX_CAST:[0-9]+]] = arith.index_cast %[[LOAD]] : i32 to index
// CHECK: %[[ADD1:[0-9]+]] = arith.addi %[[INDEX_CAST]], %[[C_NEG6]] : index
// CHECK: %[[DIV:[0-9]+]] = arith.divsi %[[ADD1]], %[[C_NEG1]] : index
// CHECK: %[[CMP:[0-9]+]] = arith.cmpi sgt, %[[DIV]], %[[C0]] : index
// CHECK: %[[SELECT:[0-9]+]] = arith.select %[[CMP]], %[[DIV]], %[[C0]] : index
// CHECK: %[[SHAPE2:[0-9]+]] = fir.shape %[[SELECT]] : (index) -> !fir.shape<1>
// CHECK: %[[SLICE:[0-9]+]] = fir.slice %[[C5]], %[[INDEX_CAST]], %[[C_NEG1]] : (index, index, index) -> !fir.slice<1>
// CHECK: %[[C1_0:.*]] = arith.constant 1 : index
// CHECK: %[[ADD2:[0-9]+]] = arith.addi %[[SELECT]], %[[C1_0]] : index
// CHECK: scf.for %[[ARG0:.*]] = %[[C1]] to %[[ADD2]] step %[[C1]] {
// CHECK:   %[[CONVERT2:.*]] = fir.convert %[[DECLARE]] : (!fir.ref<!fir.array<5xi32>>) -> memref<5xi32>
// CHECK:   %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:   %[[C1_2:.*]] = arith.constant 1 : index
// CHECK:   %[[SUB1:[0-9]+]] = arith.subi %[[ARG0]], %[[C1_2]] : index
// CHECK:   %[[MUL1:[0-9]+]] = arith.muli %[[SUB1]], %[[C_NEG1]] : index
// CHECK:   %[[SUB2:[0-9]+]] = arith.subi %[[C5]], %[[C1_1]] : index
// CHECK:   %[[ADD3:[0-9]+]] = arith.addi %[[MUL1]], %[[SUB2]] : index
// CHECK:   %[[LOAD2:[0-9]+]] = memref.load %[[CONVERT2]][%[[ADD3]]] : memref<5xi32>
func.func @array_coor_slice() attributes {fir.bindc_name = "tf4a", noinline} {
  %c0 = arith.constant 0 : index
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %c-6 = arith.constant -6 : index
  %c5 = arith.constant 5 : index
  %3 = fir.alloca i32 {adapt.valuebyref}
  %7 = fir.dummy_scope : !fir.dscope
  %10 = fir.address_of(@_QFFsECindex) : !fir.ref<!fir.array<5xi32>>
  %11 = fir.shape %c5 : (index) -> !fir.shape<1>
  %12 = fir.declare %10(%11) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFFsECindex"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<5xi32>>
  %13 = fir.declare %3 dummy_scope %7 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFFsElower"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %29 = fir.load %13 : !fir.ref<i32>
  %30 = arith.index_cast %29 : i32 to index
  %31 = arith.addi %30, %c-6 : index
  %32 = arith.divsi %31, %c-1 : index
  %33 = arith.cmpi sgt, %32, %c0 : index
  %34 = arith.select %33, %32, %c0 : index
  %35 = fir.shape %34 : (index) -> !fir.shape<1>
  %36 = fir.slice %c5, %30, %c-1 : (index, index, index) -> !fir.slice<1>
  %c1_0 = arith.constant 1 : index
  %40 = arith.addi %34, %c1_0 : index
  scf.for %arg0 = %c1 to %40 step %c1 {
    %61 = fir.array_coor %12(%11) [%36] %arg0 : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<i32>
    %62 = fir.load %61 : !fir.ref<i32>
  }
  return
}

