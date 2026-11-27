// RUN: fir-opt --lower-workdistribute %s | FileCheck %s

// An array-to-array _FortranAAssignSimple between allocatable descriptors
// (box of a heap array) must take the element-wise lowering too.

// Example Fortran code:
// real, allocatable :: x(:), y(:)
// !$omp target teams workdistribute
// y = x
// !$omp end target teams workdistribute

// CHECK-LABEL:   func.func @alloc_assign(
// CHECK:           omp.target_allocmem
// CHECK:           omp.target kernel_type
// CHECK:           omp.loop_nest
// CHECK:           %[[SRC:.*]] = fir.array_coor {{.*}} : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
// CHECK:           %[[TMP:.*]] = fir.coordinate_of {{.*}} -> !fir.ref<f32>
// CHECK:           %[[VAL:.*]] = fir.load %[[SRC]] : !fir.ref<f32>
// CHECK:           fir.store %[[VAL]] to %[[TMP]] : !fir.ref<f32>
// CHECK:           omp.target kernel_type
// CHECK:           omp.loop_nest
// CHECK:           %[[DST:.*]] = fir.array_coor {{.*}} : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
// CHECK:           %[[TMP2:.*]] = fir.coordinate_of {{.*}} -> !fir.ref<f32>
// CHECK:           %[[VAL2:.*]] = fir.load %[[TMP2]] : !fir.ref<f32>
// CHECK:           fir.store %[[VAL2]] to %[[DST]] : !fir.ref<f32>
// CHECK:           omp.target_freemem
// CHECK-NOT:       fir.call @_FortranAAssignSimple

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
func.func @alloc_assign(%x : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, %y : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
  %mapx = omp.map.info var_ptr(%x : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(tofrom) capture(ByRef) name("x") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %mapy = omp.map.info var_ptr(%y : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(tofrom) capture(ByRef) name("y") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  omp.target kernel_type(generic) map_entries(%mapx -> %argx, %mapy -> %argy : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
    omp.teams {
      %dtmp = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {pinned}
      omp.workdistribute {
        %xbox = fir.load %argx : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
        %ybox = fir.load %argy : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
        fir.store %ybox to %dtmp : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
        %str = fir.address_of(@_QQcl) : !fir.ref<!fir.char<1,2>>
        %line = arith.constant 9 : i32
        %destc = fir.convert %dtmp : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
        %srcc = fir.convert %xbox : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.box<none>
        %strc = fir.convert %str : (!fir.ref<!fir.char<1,2>>) -> !fir.ref<i8>
        fir.call @_FortranAAssignSimple(%destc, %srcc, %strc, %line) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}
func.func private @_FortranAAssignSimple(!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) attributes {fir.runtime}
fir.global linkonce @_QQcl constant : !fir.char<1,2> {
  %0 = fir.string_lit "f\00"(2) : !fir.char<1,2>
  fir.has_value %0 : !fir.char<1,2>
}
}
