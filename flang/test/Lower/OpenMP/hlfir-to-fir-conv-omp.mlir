// Tests HLFIR-to-FIR conversion aspects relevant to OpenMP. For example, that
// the correct alloca block is chosen for OMP regions.

// RUN: fir-opt --convert-hlfir-to-fir %s -o - | \
// RUN: FileCheck %s

fir.global internal @_QQro.1xi4.0(dense<42> : tensor<1xi32>) constant : !fir.array<1xi32>

func.func @_QPfoo() {
  %c1 = arith.constant 1 : index
  %host_alloc = fir.alloca !fir.array<1xi32> {bindc_name = "arr", uniq_name = "_QFfooEarr"}

  %1 = fir.shape %c1 : (index) -> !fir.shape<1>
  %host_decl:2 = hlfir.declare %host_alloc(%1) {uniq_name = "_QFfooEarr"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
  %map_info = omp.map.info var_ptr(%host_decl#1 : !fir.ref<!fir.array<1xi32>>, !fir.array<1xi32>) map_clauses(implicit, tofrom) capture(ByRef)  -> !fir.ref<!fir.array<1xi32>> {name = "arr"}

  // CHECK: omp.target
  omp.target map_entries(%map_info -> %arg1 : !fir.ref<!fir.array<1xi32>>)  {
    %c1_2 = arith.constant 1 : index
    %21 = fir.shape %c1_2 : (index) -> !fir.shape<1>

    // CHECK: %[[TARGET_DECL:.*]] = fir.declare
    %target_decl:2 = hlfir.declare %arg1(%21) {uniq_name = "_QFfooEarr"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)

    // CHECK: omp.teams
    omp.teams {
      %c1_3 = arith.constant 1 : i32
      %c10 = arith.constant 10 : i32

      // CHECK: omp.parallel
      omp.parallel {
        // CHECK: %[[TO_BOX_ALLOC:.*]] = fir.alloca !fir.box<!fir.array<1xi32>> {pinned}
        // CHECK: omp.distribute
        omp.distribute {
          // CHECK: omp.wsloop
          omp.wsloop {
            // CHECK: omp.loop_nest
            omp.loop_nest (%arg2) : i32 = (%c1_3) to (%c10) inclusive step (%c1_3) {
              %25 = fir.address_of(@_QQro.1xi4.0) : !fir.ref<!fir.array<1xi32>>
              %26 = fir.shape %c1_2 : (index) -> !fir.shape<1>
              %27:2 = hlfir.declare %25(%26) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.1xi4.0"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)


              // CHECK: %[[EMBOX:.*]] = fir.embox %[[TARGET_DECL]]
              // CHECK: fir.store %[[EMBOX]] to %[[TO_BOX_ALLOC]]
              // CHECK: %[[BOX_ALLOC_CONV:.*]] = fir.convert %[[TO_BOX_ALLOC]] : (!fir.ref<!fir.box<!fir.array<1xi32>>>) -> !fir.ref<!fir.box<none>>
              // CHECK: fir.call @_FortranAAssign(%[[BOX_ALLOC_CONV]], {{.*}})
              hlfir.assign %27#0 to %target_decl#0 : !fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>
              // CHECK: omp.yield
              omp.yield
            }
          } {omp.composite}
        } {omp.composite}
        // CHECK: omp.terminator
        omp.terminator
      } {omp.composite}
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}
