// RUN: fir-opt --lower-workdistribute %s | FileCheck %s

// An array-to-array _FortranAAssign in target teams workdistribute must lower
// to an element-wise fir.array_coor copy, not a flat omp_target_memcpy.

// Example Fortran code:
// !$omp target teams workdistribute
// a(:,:) = b(:,:)
// !$omp end target teams workdistribute

// CHECK-LABEL:   func.func @array_assign(
// CHECK:           omp.target_data
// CHECK:           omp.target
// CHECK:             omp.teams
// CHECK:               omp.parallel
// CHECK:                 omp.distribute
// CHECK:                   omp.wsloop
// CHECK:                     omp.loop_nest
// CHECK:                       %[[SRC:.*]] = fir.array_coor {{.*}} : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
// CHECK:                       %[[VAL:.*]] = fir.load %[[SRC]] : !fir.ref<f32>
// CHECK:                       %[[DST:.*]] = fir.array_coor {{.*}} : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
// CHECK:                       fir.store %[[VAL]] to %[[DST]] : !fir.ref<f32>
// CHECK-NOT:         omp_target_memcpy

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
func.func @array_assign(%a : !fir.ref<!fir.array<?x?xf32>>, %b : !fir.ref<!fir.array<?x?xf32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %ub0 = arith.subi %c10, %c1 : index
  %bnd0 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%ub0 : index) extent(%c10 : index) stride(%c1 : index) start_idx(%c1 : index)
  %ub1 = arith.subi %c20, %c1 : index
  %bnd1 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%ub1 : index) extent(%c20 : index) stride(%c1 : index) start_idx(%c1 : index)
  %mapa = omp.map.info var_ptr(%a : !fir.ref<!fir.array<?x?xf32>>, f32) map_clauses(implicit, tofrom) capture(ByRef) bounds(%bnd0, %bnd1) name("a") -> !fir.ref<!fir.array<?x?xf32>>
  %mapb = omp.map.info var_ptr(%b : !fir.ref<!fir.array<?x?xf32>>, f32) map_clauses(implicit, tofrom) capture(ByRef) bounds(%bnd0, %bnd1) name("b") -> !fir.ref<!fir.array<?x?xf32>>
  omp.target kernel_type(generic) map_entries(%mapa -> %arga, %mapb -> %argb : !fir.ref<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>) {
    // omp.target is isolated from above, so re-declare the extents here.
    %e0 = arith.constant 10 : index
    %e1 = arith.constant 20 : index
    %shape = fir.shape %e0, %e1 : (index, index) -> !fir.shape<2>
    %da = fir.declare %arga(%shape) {uniq_name = "a"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.ref<!fir.array<?x?xf32>>
    %db = fir.declare %argb(%shape) {uniq_name = "b"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.ref<!fir.array<?x?xf32>>
    omp.teams {
      %dtmp = fir.alloca !fir.box<!fir.array<?x?xf32>> {pinned}
      omp.workdistribute {
        %srcbox = fir.embox %db(%shape) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
        %dstbox = fir.embox %da(%shape) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
        fir.store %dstbox to %dtmp : !fir.ref<!fir.box<!fir.array<?x?xf32>>>
        %str = fir.address_of(@_QQcl) : !fir.ref<!fir.char<1,2>>
        %line = arith.constant 13 : i32
        %destc = fir.convert %dtmp : (!fir.ref<!fir.box<!fir.array<?x?xf32>>>) -> !fir.ref<!fir.box<none>>
        %srcc = fir.convert %srcbox : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
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
