// RUN: fir-opt --lower-workdistribute %s | FileCheck %s

// An array-to-array _FortranAAssign whose sections overlap must copy through a
// temporary. The copy is split into two kernels so every read of the source
// finishes before any write to the destination.

// Example Fortran code:
// !$omp target teams workdistribute
// a(2:10) = a(1:9)
// !$omp end target teams workdistribute

// CHECK-LABEL:   func.func @overlap_assign(
// CHECK:           omp.target_data
// CHECK:           omp.target_allocmem
// CHECK:           omp.target kernel_type
// CHECK:           omp.loop_nest
// CHECK:           %[[SRC:.*]] = fir.array_coor
// CHECK:           %[[TMP:.*]] = fir.coordinate_of
// CHECK:           %[[VAL:.*]] = fir.load %[[SRC]]
// CHECK:           fir.store %[[VAL]] to %[[TMP]]
// CHECK:           omp.target kernel_type
// CHECK:           omp.loop_nest
// CHECK:           %[[DST:.*]] = fir.array_coor
// CHECK:           %[[TMP2:.*]] = fir.coordinate_of
// CHECK:           %[[VAL2:.*]] = fir.load %[[TMP2]]
// CHECK:           fir.store %[[VAL2]] to %[[DST]]
// CHECK:           omp.target_freemem
// CHECK-NOT:       fir.call @_FortranAAssign

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
func.func @overlap_assign(%a : !fir.ref<!fir.array<?xf32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %ub0 = arith.subi %c10, %c1 : index
  %bnd0 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%ub0 : index) extent(%c10 : index) stride(%c1 : index) start_idx(%c1 : index)
  %mapa = omp.map.info var_ptr(%a : !fir.ref<!fir.array<?xf32>>, f32) map_clauses(implicit, tofrom) capture(ByRef) bounds(%bnd0) name("a") -> !fir.ref<!fir.array<?xf32>>
  omp.target kernel_type(generic) map_entries(%mapa -> %arga : !fir.ref<!fir.array<?xf32>>) {
    %e0 = arith.constant 10 : index
    %shape = fir.shape %e0 : (index) -> !fir.shape<1>
    %da = fir.declare %arga(%shape) uniq_name("a") : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf32>>
    omp.teams {
      %dtmp = fir.alloca !fir.box<!fir.array<?xf32>> {pinned}
      omp.workdistribute {
        %one = arith.constant 1 : index
        %two = arith.constant 2 : index
        %nine = arith.constant 9 : index
        %srcslice = fir.slice %one, %nine, %one : (index, index, index) -> !fir.slice<1>
        %dstslice = fir.slice %two, %e0, %one : (index, index, index) -> !fir.slice<1>
        %srcbox = fir.embox %da(%shape) [%srcslice] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
        %dstbox = fir.embox %da(%shape) [%dstslice] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
        fir.store %dstbox to %dtmp : !fir.ref<!fir.box<!fir.array<?xf32>>>
        %str = fir.address_of(@_QQcl) : !fir.ref<!fir.char<1,2>>
        %line = arith.constant 9 : i32
        %destc = fir.convert %dtmp : (!fir.ref<!fir.box<!fir.array<?xf32>>>) -> !fir.ref<!fir.box<none>>
        %srcc = fir.convert %srcbox : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
        %strc = fir.convert %str : (!fir.ref<!fir.char<1,2>>) -> !fir.ref<i8>
        fir.call @_FortranAAssign(%destc, %srcc, %strc, %line) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}
func.func private @_FortranAAssign(!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) attributes {fir.runtime}
fir.global linkonce @_QQcl constant : !fir.char<1,2> {
  %0 = fir.string_lit "f\00"(2) : !fir.char<1,2>
  fir.has_value %0 : !fir.char<1,2>
}
}
