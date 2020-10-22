! RUN: bbc -emit-fir -o - %s | FileCheck %s

! Check more advanced equivalence cases

! Several set of local and global equivalences in the same scope
! CHECK-LABEL: @_QPtest_eq_sets
subroutine test_eq_sets
  DIMENSION Al(4), Bl(4)
  EQUIVALENCE (Al(1), Bl(2))
  ! CHECK-DAG: %[[albl:.*]] = fir.alloca !fir.array<20xi8>
  ! CHECK-DAG: %[[alAddr:.*]] = fir.coordinate_of %[[albl]], %c4{{.*}} : (!fir.ref<!fir.array<20xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[al:.*]] = fir.convert %[[alAddr]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<4xf32>>
  ! CHECK-DAG: %[[blAddr:.*]] = fir.coordinate_of %[[albl]], %c0{{.*}} : (!fir.ref<!fir.array<20xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[bl:.*]] = fir.convert %[[blAddr]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<4xf32>>

  DIMENSION Il(2), Xl(2)
  EQUIVALENCE (Il(2), Xl(1))
  ! CHECK-DAG: %[[ilxl:.*]] = fir.alloca !fir.array<12xi8>
  ! CHECK-DAG: %[[ilAddr:.*]] = fir.coordinate_of %[[ilxl]], %c0{{.*}} : (!fir.ref<!fir.array<12xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[il:.*]] = fir.convert %[[ilAddr]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2xi32>>
  ! CHECK-DAG: %[[xlAddr:.*]] = fir.coordinate_of %[[ilxl]], %c4{{.*}} : (!fir.ref<!fir.array<12xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[xl:.*]] = fir.convert %[[xlAddr]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2xf32>>

  DIMENSION Ag(2), Bg(2)
  SAVE Ag, Bg
  EQUIVALENCE (Ag(1), Bg(2))
  ! CHECK-DAG: %[[agbgStore:.*]] = fir.address_of(@_QFtest_eq_setsEag) : !fir.ref<tuple<!fir.array<4xi8>, !fir.array<8xi8>>>
  ! CHECK-DAG: %[[agbg:.*]] = fir.convert %[[agbgStore]] : (!fir.ref<tuple<!fir.array<4xi8>, !fir.array<8xi8>>>) -> !fir.ref<!fir.array<12xi8>>
  ! CHECK-DAG: %[[agAddr:.*]] = fir.coordinate_of %[[agbg]], %c4{{.*}} : (!fir.ref<!fir.array<12xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[ag:.*]] = fir.convert %[[agAddr]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2xf32>>
  ! CHECK-DAG: %[[bgAddr:.*]] = fir.coordinate_of %[[agbg]], %c0{{.*}} : (!fir.ref<!fir.array<12xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[bg:.*]] = fir.convert %[[bgAddr]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2xf32>>

  DIMENSION Ig(2), Xg(2)
  SAVE Ig, Xg
  EQUIVALENCE (Ig(1), Xg(1))
  ! CHECK-DAG: %[[igxgStore:.*]] = fir.address_of(@_QFtest_eq_setsEig) : !fir.ref<tuple<!fir.array<8xi8>>>
  ! CHECK-DAG: %[[igxg:.*]] = fir.convert %[[igxgStore]] : (!fir.ref<tuple<!fir.array<8xi8>>>) -> !fir.ref<!fir.array<8xi8>>
  ! CHECK-DAG: %[[igOffset:.*]] = constant 0 : index
  ! CHECK-DAG: %[[igAddr:.*]] = fir.coordinate_of %[[igxg]], %c0{{.*}} : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[ig:.*]] = fir.convert %[[igAddr]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2xi32>>
  ! CHECK-DAG: %[[xgAddr:.*]] = fir.coordinate_of %[[igxg]], %c0{{.*}} : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[xg:.*]] = fir.convert %[[xgAddr]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2xf32>>

  call fooc(Al, Bl, Il, Xl, Ag, Bg, Xg, Ig)
  ! CHECK: fir.call @_QPfooc(%[[al]], %[[bl]], %[[il]], %[[xl]], %[[ag]], %[[bg]], %[[xg]], %[[ig]])

end subroutine


! Mixing global equivalence and entry
! CHECK-LABEL: @_QPeq_and_entry_foo()
subroutine eq_and_entry_foo
  SAVE x, i
  DIMENSION :: x(2)
  EQUIVALENCE (x(2), i) 
  call foo1(x, i)
  ! CHECK: %[[xiStore:.*]] = fir.address_of(@_QFeq_and_entry_fooEi) : !fir.ref<tuple<!fir.array<4xi8>, !fir.array<4xi8>>>
  ! CHECK-DAG: %[[xi:.*]] = fir.convert %[[xiStore]] : (!fir.ref<tuple<!fir.array<4xi8>, !fir.array<4xi8>>>) -> !fir.ref<!fir.array<8xi8>>

  ! CHECK-DAG: %[[iOffset:.*]] = constant 4 : index
  ! CHECK-DAG: %[[iAddr:.*]] = fir.coordinate_of %[[xi]], %[[iOffset]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[i:.*]] = fir.convert %[[iAddr]] : (!fir.ref<i8>) -> !fir.ref<i32>

  ! CHECK-DAG: %[[xOffset:.*]] = constant 0 : index
  ! CHECK-DAG: %[[xAddr:.*]] = fir.coordinate_of %[[xi]], %[[xOffset]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[x:.*]] = fir.convert %[[xAddr]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2xf32>>
  call foo2(x, i)
  ! CHECK: fir.call @_QPfoo1(%[[x]], %[[i]]) : (!fir.ref<!fir.array<2xf32>>, !fir.ref<i32>) -> ()
  entry eq_and_entry_bar
  call foo2(x, i)
  ! CHECK: fir.call @_QPfoo2(%[[x]], %[[i]]) : (!fir.ref<!fir.array<2xf32>>, !fir.ref<i32>) -> ()
end

! CHECK-LABEL: @_QPeq_and_entry_bar()
  ! CHECK: %[[xiStore:.*]] = fir.address_of(@_QFeq_and_entry_fooEi) : !fir.ref<tuple<!fir.array<4xi8>, !fir.array<4xi8>>>
  ! CHECK-DAG: %[[xi:.*]] = fir.convert %[[xiStore]] : (!fir.ref<tuple<!fir.array<4xi8>, !fir.array<4xi8>>>) -> !fir.ref<!fir.array<8xi8>>

  ! CHECK-DAG: %[[iOffset:.*]] = constant 4 : index
  ! CHECK-DAG: %[[iAddr:.*]] = fir.coordinate_of %[[xi]], %[[iOffset]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[i:.*]] = fir.convert %[[iAddr]] : (!fir.ref<i8>) -> !fir.ref<i32>

  ! CHECK-DAG: %[[xOffset:.*]] = constant 0 : index
  ! CHECK-DAG: %[[xAddr:.*]] = fir.coordinate_of %[[xi]], %[[xOffset]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
  ! CHECK-DAG: %[[x:.*]] = fir.convert %[[xAddr]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2xf32>>
  ! CHECK-NOT: fir.call @_QPfoo1
  ! CHECK: fir.call @_QPfoo2(%[[x]], %[[i]]) : (!fir.ref<!fir.array<2xf32>>, !fir.ref<i32>) -> ()

