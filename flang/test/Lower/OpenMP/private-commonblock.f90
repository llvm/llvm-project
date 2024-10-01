! RUN: %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -mmlir --openmp-enable-delayed-privatization=true -o - %s 2>&1 \
! RUN: | FileCheck %s

!CHECK: func.func @_QPprivate_common() {
!CHECK: omp.parallel private(@{{.*}} %{{.*}}#0 -> %[[X:.*]], @{{.*}} %{{.*}}#0 -> %[[Y:.*]] : {{.*}}) {
!CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFprivate_commonEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFprivate_commonEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK: omp.terminator
!CHECK: }
!CHECK: return
!CHECK: }
subroutine private_common
  common /c/ x, y
  real x, y
  !$omp parallel private(/c/)
  !$omp end parallel
end subroutine

!CHECK:    %[[BLK_ADDR:.*]] = fir.address_of(@blk_) : !fir.ref<!fir.array<74xi8>>
!CHECK:    %[[I8_ARR:.*]] = fir.convert %[[BLK_ADDR]] : (!fir.ref<!fir.array<74xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK:    %[[C0:.*]] = arith.constant 0 : index
!CHECK:    %[[A_I8_REF:.*]] = fir.coordinate_of %[[I8_ARR]], %[[C0]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK:    %[[A_REF:.*]] = fir.convert %[[A_I8_REF]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK:    %[[A_DECL:.*]]:2 = hlfir.declare %[[A_REF]] {uniq_name = "_QFprivate_clause_commonblockEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[I8_ARR:.*]] = fir.convert %[[BLK_ADDR]] : (!fir.ref<!fir.array<74xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK:    %[[C4:.*]] = arith.constant 4 : index
!CHECK:    %[[B_I8_REF:.*]] = fir.coordinate_of %[[I8_ARR]], %[[C4]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK:    %[[B_REF:.*]] = fir.convert %[[B_I8_REF:.*]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<10xf32>>
!CHECK:    %[[C10:.*]] = arith.constant 10 : index
!CHECK:    %[[SH10:.*]] = fir.shape %[[C10]] : (index) -> !fir.shape<1>
!CHECK:    %[[B_DECL:.*]]:2 = hlfir.declare %[[B_REF]](%[[SH10]]) {uniq_name = "_QFprivate_clause_commonblockEb"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
!CHECK:    %[[I8_ARR:.*]] = fir.convert %[[BLK_ADDR]] : (!fir.ref<!fir.array<74xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK:    %[[C44:.*]] = arith.constant 44 : index
!CHECK:    %[[C_I8_REF:.*]] = fir.coordinate_of %[[I8_ARR]], %[[C44]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK:    %[[C_REF:.*]] = fir.convert %[[C_I8_REF]] : (!fir.ref<i8>) -> !fir.ref<!fir.char<1,5>>
!CHECK:    %[[C5:.*]] = arith.constant 5 : index
!CHECK:    %[[C_DECL:.*]]:2 = hlfir.declare %[[C_REF]] typeparams %[[C5]] {uniq_name = "_QFprivate_clause_commonblockEc"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
!CHECK:    %[[I8_ARR:.*]] = fir.convert %[[BLK_ADDR]] : (!fir.ref<!fir.array<74xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK:    %[[C49:.*]] = arith.constant 49 : index
!CHECK:    %[[D_I8_REF:.*]] = fir.coordinate_of %[[I8_ARR]], %[[C49]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK:    %[[D_REF:.*]] = fir.convert %[[D_I8_REF]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<5x!fir.char<1,5>>>
!CHECK:    %[[TP5:.*]] = arith.constant 5 : index
!CHECK:    %[[C5:.*]] = arith.constant 5 : index
!CHECK:    %[[SH5:.*]] = fir.shape %[[C5]] : (index) -> !fir.shape<1>
!CHECK:    %[[D_DECL:.*]]:2 = hlfir.declare %[[D_REF]](%[[SH5:.*]]) typeparams %[[TP5]] {uniq_name = "_QFprivate_clause_commonblockEd"} : (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.ref<!fir.array<5x!fir.char<1,5>>>)
!CHECK:    %[[C_BOX:.*]] = fir.emboxchar %[[C_DECL]]#1, %c5 : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
!CHECK:    %[[D_REF:.*]] = fir.convert %[[D_DECL]]#1 : (!fir.ref<!fir.array<5x!fir.char<1,5>>>) -> !fir.ref<!fir.char<1,5>>
!CHECK:    %[[D_BOX:.*]] = fir.emboxchar %[[D_REF]], %[[TP5]] : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
!CHECK:    fir.call @_QPsub1(%[[A_DECL]]#1, %[[B_DECL]]#1, %[[C_BOX]], %[[D_BOX]]) fastmath<contract> : (!fir.ref<i32>, !fir.ref<!fir.array<10xf32>>, !fir.boxchar<1>, !fir.boxchar<1>) -> ()
!CHECK:    omp.parallel private(@{{.*}} %{{.*}}#0 -> %[[A_PVT_REF:.*]], @{{.*}} %{{.*}}#0 -> %[[B_PVT_REF:.*]], @{{.*}} %{{.*}}#0 -> %[[C_PVT_REF:.*]], @{{.*}} %{{.*}}#0 -> %[[D_PVT_REF:.*]] : {{.*}}) {
!CHECK:      %[[A_PVT_DECL:.*]]:2 = hlfir.declare %[[A_PVT_REF]] {uniq_name = "_QFprivate_clause_commonblockEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:      %[[SH10:.*]] = fir.shape %c10{{.*}} : (index) -> !fir.shape<1>
!CHECK:      %[[B_PVT_DECL:.*]]:2 = hlfir.declare %[[B_PVT_REF]](%[[SH10]]) {uniq_name = "_QFprivate_clause_commonblockEb"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
!CHECK:      %[[C_PVT_DECL:.*]]:2 = hlfir.declare %[[C_PVT_REF]] typeparams %{{.*}} {uniq_name = "_QFprivate_clause_commonblockEc"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
!CHECK:      %[[SH5:.*]] = fir.shape %c5{{.*}} : (index) -> !fir.shape<1>
!CHECK:      %[[D_PVT_DECL:.*]]:2 = hlfir.declare %[[D_PVT_REF]](%[[SH5]]) typeparams %c5{{.*}} {uniq_name = "_QFprivate_clause_commonblockEd"} : (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.ref<!fir.array<5x!fir.char<1,5>>>)
!CHECK:      %[[C_PVT_BOX:.*]] = fir.emboxchar %[[C_PVT_DECL]]#1, %{{.*}} : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
!CHECK:      %[[D_PVT_REF:.*]] = fir.convert %[[D_PVT_DECL]]#1 : (!fir.ref<!fir.array<5x!fir.char<1,5>>>) -> !fir.ref<!fir.char<1,5>>
!CHECK:      %[[D_PVT_BOX:.*]] = fir.emboxchar %[[D_PVT_REF]], %{{.*}} : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
!CHECK:      fir.call @_QPsub2(%[[A_PVT_DECL]]#1, %[[B_PVT_DECL]]#1, %[[C_PVT_BOX]], %[[D_PVT_BOX]]) fastmath<contract> : (!fir.ref<i32>, !fir.ref<!fir.array<10xf32>>, !fir.boxchar<1>, !fir.boxchar<1>) -> ()
!CHECK:      omp.terminator
!CHECK:    }
!CHECK:    %[[C_BOX:.*]] = fir.emboxchar %[[C_DECL]]#1, %{{.*}} : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
!CHECK:    %[[D_REF:.*]] = fir.convert %[[D_DECL]]#1 : (!fir.ref<!fir.array<5x!fir.char<1,5>>>) -> !fir.ref<!fir.char<1,5>>
!CHECK:    %[[D_BOX:.*]] = fir.emboxchar %[[D_REF]], %{{.*}} : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
!CHECK:    fir.call @_QPsub3(%[[A_DECL]]#1, %[[B_DECL]]#1, %[[C_BOX]], %[[D_BOX]]) fastmath<contract> : (!fir.ref<i32>, !fir.ref<!fir.array<10xf32>>, !fir.boxchar<1>, !fir.boxchar<1>) -> ()
subroutine private_clause_commonblock()
  integer::a
  real::b(10)
  character(5):: c, d(5)
  common /blk/ a, b, c, d
  
  call sub1(a, b, c, d)
  !$omp parallel private(/blk/)
        call sub2(a, b, c, d)
  !$omp end parallel
  call sub3(a, b, c, d)
end subroutine

!CHECK: func.func @_QPprivate_clause_commonblock_pointer() {
!CHECK:    %[[BLK_ADDR:.*]] = fir.address_of(@blk_) : !fir.ref<!fir.array<74xi8>>
!CHECK:    %[[BLK_I8_REF:.*]] = fir.convert %[[BLK_ADDR]] : (!fir.ref<!fir.array<74xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK:    %[[C24:.*]] = arith.constant 24 : index
!CHECK:    %[[A_I8_REF:.*]] = fir.coordinate_of %[[BLK_I8_REF]], %[[C24]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK:    %[[A_REF:.*]] = fir.convert %[[A_I8_REF]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK:    %[[A_DECL:.*]]:2 = hlfir.declare %[[A_REF]] {uniq_name = "_QFprivate_clause_commonblock_pointerEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[BLK_I8_REF:.*]] = fir.convert %[[BLK_ADDR]] : (!fir.ref<!fir.array<74xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK:    %[[C0:.*]] = arith.constant 0 : index
!CHECK:    %[[C_I8_REF:.*]] = fir.coordinate_of %[[BLK_I8_REF]], %[[C0]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK:    %[[C_REF:.*]] = fir.convert %[[C_I8_REF]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!CHECK:    %[[C_DECL:.*]]:2 = hlfir.declare %[[C_REF]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFprivate_clause_commonblock_pointerEc"} : (!fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>)
!CHECK:    %[[C_BOX:.*]] = fir.load %[[C_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!CHECK:    %[[C_ADDR:.*]] = fir.box_addr %[[C_BOX]] : (!fir.box<!fir.ptr<!fir.complex<4>>>) -> !fir.ptr<!fir.complex<4>>
!CHECK:    %[[C_REF:.*]] = fir.convert %[[C_ADDR]] : (!fir.ptr<!fir.complex<4>>) -> !fir.ref<!fir.complex<4>>
!CHECK:    fir.call @_QPsub4(%[[C_REF]], %[[A_DECL]]#1) fastmath<contract> : (!fir.ref<!fir.complex<4>>, !fir.ref<i32>) -> ()
!CHECK: omp.parallel private(@{{.*}} %{{.*}}#0 -> %[[C_PVT_REF:.*]], @{{.*}} %{{.*}}#0 -> %[[A_PVT_REF:.*]] : {{.*}}) {
!CHECK:      %[[C_PVT_DECL:.*]]:2 = hlfir.declare %[[C_PVT_REF]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFprivate_clause_commonblock_pointerEc"} : (!fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>)
!CHECK:      %[[A_PVT_DECL:.*]]:2 = hlfir.declare %[[A_PVT_REF]] {uniq_name = "_QFprivate_clause_commonblock_pointerEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:      %[[C_PVT_BOX:.*]] = fir.load %[[C_PVT_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!CHECK:      %[[C_PVT_ADDR:.*]] = fir.box_addr %[[C_PVT_BOX]] : (!fir.box<!fir.ptr<!fir.complex<4>>>) -> !fir.ptr<!fir.complex<4>>
!CHECK:      %[[C_PVT_REF:.*]] = fir.convert %[[C_PVT_ADDR]] : (!fir.ptr<!fir.complex<4>>) -> !fir.ref<!fir.complex<4>>
!CHECK:      fir.call @_QPsub5(%[[C_PVT_REF]], %[[A_PVT_DECL]]#1) fastmath<contract> : (!fir.ref<!fir.complex<4>>, !fir.ref<i32>) -> ()
!CHECK:      omp.terminator
!CHECK:    }
!CHECK:    %[[C_BOX:.*]] = fir.load %[[C_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!CHECK:    %[[C_ADDR:.*]] = fir.box_addr %[[C_BOX]] : (!fir.box<!fir.ptr<!fir.complex<4>>>) -> !fir.ptr<!fir.complex<4>>
!CHECK:    %[[C_REF:.*]] = fir.convert %[[C_ADDR]] : (!fir.ptr<!fir.complex<4>>) -> !fir.ref<!fir.complex<4>>
!CHECK:    fir.call @_QPsub6(%[[C_REF]], %[[A_DECL]]#1) fastmath<contract> : (!fir.ref<!fir.complex<4>>, !fir.ref<i32>) -> ()
!CHECK: return
!CHECK: }
subroutine private_clause_commonblock_pointer()
  complex, pointer :: c
  integer:: a
  common /blk/ c, a
  call sub4(c, a)
  !$omp parallel private(/blk/)
        call sub5(c, a)
  !$omp end parallel
  call sub6(c, a)
end subroutine
