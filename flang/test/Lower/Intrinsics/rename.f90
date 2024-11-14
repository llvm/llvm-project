!RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

!CHECK-LABEL: func.func @_QPtest_rename
!CHECK-SAME:    %[[dummySrc:.*]]: !fir.boxchar<1> {fir.bindc_name = "src"},
!CHECK-SAME:    %[[dummyDst:.*]]: !fir.boxchar<1> {fir.bindc_name = "dst"}) {
subroutine test_rename(src, dst)
    implicit none
    character(*) :: src, dst

    call rename(src, dst)
    !CHECK:      %[[dstUnbox:.*]]:2 = fir.unboxchar %[[dummyDst]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    !CHECK-NEXT: %[[dstDecl:.*]]:2 = hlfir.declare %[[dstUnbox]]#0 typeparams %[[dstUnbox]]#1 dummy_scope %0 {uniq_name = "_QFtest_renameEdst"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    !CHECK-NEXT: %[[srcUnbox:.*]]:2 = fir.unboxchar %[[dummySrc]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    !CHECK-NEXT: %[[srcDecl:.*]]:2 = hlfir.declare %3#0 typeparams %[[srcUnbox]]#1 dummy_scope %0 {uniq_name = "_QFtest_renameEsrc"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    !CHECK-NEXT: %[[srcBox:.*]] = fir.embox %[[srcDecl]]#1 typeparams %[[srcUnbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
    !CHECK-NEXT: %[[dstBox:.*]] = fir.embox %[[dstDecl]]#1 typeparams %[[dstUnbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
    !CHECK-NEXT: %[[statusBox:.*]] = fir.absent !fir.box<none>
    !CHECK-NEXT: %[[sourceFile:.*]] = fir.address_of(@[[someString:.*]]) : !fir.ref<!fir.char<1,[[len:.*]]>>
    !CHECK-NEXT: %[[c10_i32:.*]] = arith.constant [[line:.*]] : i32
    !CHECK-NEXT: %[[src:.*]] = fir.convert %[[srcBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
    !CHECK-NEXT: %[[dst:.*]] = fir.convert %[[dstBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
    !CHECK-NEXT: %[[loc:.*]] = fir.convert %[[sourceFileConv:.*]]: (!fir.ref<!fir.char<1,[[len:.*]]>>) -> !fir.ref<i8>
    !CHECK-NEXT: %[[result:.*]] = fir.call @_FortranARename(%[[src]], %[[dst]], %[[statusBox]], %[[loc]], %[[c10_i32]]) fastmath<contract> : (!fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
end subroutine test_rename

!CHECK-LABEL: func.func @_QPtest_rename_status
!CHECK-SAME:    %[[dummySrc:.*]]: !fir.boxchar<1> {fir.bindc_name = "src"},
!CHECK-SAME:    %[[dummyDst:.*]]: !fir.boxchar<1> {fir.bindc_name = "dst"}) {
subroutine test_rename_status(src, dst)
    implicit none
    character(*) :: src, dst
    integer :: status

    call rename(src, dst, status)
    !CHECK:      %[[dstUnbox:.*]]:2 = fir.unboxchar %[[dummyDst]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    !CHECK-NEXT: %[[dstDecl:.*]]:2 = hlfir.declare %[[dstUnbox]]#0 typeparams %[[dstUnbox]]#1 dummy_scope %0 {uniq_name = "_QFtest_rename_statusEdst"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    !CHECK-NEXT: %[[srcUnbox:.*]]:2 = fir.unboxchar %[[dummySrc]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    !CHECK-NEXT: %[[srcDecl:.*]]:2 = hlfir.declare %3#0 typeparams %[[srcUnbox]]#1 dummy_scope %0 {uniq_name = "_QFtest_rename_statusEsrc"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    !CHECK-NEXT: %[[statusAlloc:.*]] = fir.alloca i32 {bindc_name = "status", uniq_name = "_QFtest_rename_statusEstatus"}
    !CHECK-NEXT: %[[statusDecl:.*]]:2 = hlfir.declare %[[statusAlloc]] {uniq_name = "_QFtest_rename_statusEstatus"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    !CHECK-NEXT: %[[srcBox:.*]] = fir.embox %[[srcDecl]]#1 typeparams %[[srcUnbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
    !CHECK-NEXT: %[[dstBox:.*]] = fir.embox %[[dstDecl]]#1 typeparams %[[dstUnbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
    !CHECK-NEXT: %[[statusBox:.*]] = fir.embox %[[statusDecl]]#1 : (!fir.ref<i32>) -> !fir.box<i32>
    !CHECK-NEXT: %[[sourceFile:.*]] = fir.address_of(@[[someString:.*]]) : !fir.ref<!fir.char<1,[[len:.*]]>>
    !CHECK-NEXT: %[[c10_i32:.*]] = arith.constant [[line:.*]] : i32
    !CHECK-NEXT: %[[src:.*]] = fir.convert %[[srcBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
    !CHECK-NEXT: %[[dst:.*]] = fir.convert %[[dstBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
    !CHECK-NEXT: %[[status:.*]] = fir.convert %[[statusBox]] : (!fir.box<i32>) -> !fir.box<none>
    !CHECK-NEXT: %[[loc:.*]] = fir.convert %[[sourceFileConv:.*]]: (!fir.ref<!fir.char<1,[[len:.*]]>>) -> !fir.ref<i8>
    !CHECK-NEXT: %[[result:.*]] = fir.call @_FortranARename(%[[src]], %[[dst]], %[[status]], %[[loc]], %[[c10_i32]]) fastmath<contract> : (!fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
end subroutine test_rename_status
