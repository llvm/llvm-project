// RUN: fir-opt %s --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-data)" -split-input-file | FileCheck %s

// A common block that is used in a compute region only as the `storage`
// operand of the fir.declare of its members must not be implicitly mapped as a
// whole: that would clash with explicit mappings of the individual members.
//
//   integer(1) :: r1(1000)
//   real(4) :: r2(1000)
//   common /com/ r1, r2
//   !$acc enter data create(r1, r2)
//   !$acc parallel present(r1, r2)
//     r1 = 1
//   !$acc end parallel

fir.global common @com_(dense<0> : vector<5000xi8>) : !fir.array<5000xi8>

func.func @members_only_via_storage() {
  %c0 = arith.constant 0 : index
  %c1000 = arith.constant 1000 : index
  %c1_i8 = arith.constant 1 : i8
  %com = fir.address_of(@com_) : !fir.ref<!fir.array<5000xi8>>
  %shape = fir.shape %c1000 : (index) -> !fir.shape<1>
  %p1 = fir.coordinate_of %com, %c0 : (!fir.ref<!fir.array<5000xi8>>, index) -> !fir.ref<i8>
  %r1p = fir.convert %p1 : (!fir.ref<i8>) -> !fir.ref<!fir.array<1000xi8>>
  %r1 = fir.declare %r1p(%shape) storage(%com[0]) uniq_name("_QFEr1") : (!fir.ref<!fir.array<1000xi8>>, !fir.shape<1>, !fir.ref<!fir.array<5000xi8>>) -> !fir.ref<!fir.array<1000xi8>>
  %c1 = acc.create varPtr(%r1 : !fir.ref<!fir.array<1000xi8>>) structured(false) name("r1") -> !fir.ref<!fir.array<1000xi8>>
  acc.enter_data dataOperands(%c1 : !fir.ref<!fir.array<1000xi8>>)
  %pr1 = acc.present varPtr(%r1 : !fir.ref<!fir.array<1000xi8>>) name("r1") -> !fir.ref<!fir.array<1000xi8>>
  acc.parallel dataOperands(%pr1 : !fir.ref<!fir.array<1000xi8>>) {
    %d = fir.declare %pr1(%shape) storage(%com[0]) uniq_name("_QFEr1") : (!fir.ref<!fir.array<1000xi8>>, !fir.shape<1>, !fir.ref<!fir.array<5000xi8>>) -> !fir.ref<!fir.array<1000xi8>>
    %e = fir.array_coor %d(%shape) %c1000 : (!fir.ref<!fir.array<1000xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
    fir.store %c1_i8 to %e : !fir.ref<i8>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @members_only_via_storage
// CHECK-NOT: name("com_")
// CHECK: acc.parallel

// -----

// If the whole common block is accessed directly in the region (not only as
// the storage of a declare), it must still be implicitly mapped.

fir.global common @com_(dense<0> : vector<5000xi8>) : !fir.array<5000xi8>

func.func @whole_block_accessed() {
  %c0 = arith.constant 0 : index
  %c1_i8 = arith.constant 1 : i8
  %com = fir.address_of(@com_) : !fir.ref<!fir.array<5000xi8>>
  acc.parallel {
    %p = fir.coordinate_of %com, %c0 : (!fir.ref<!fir.array<5000xi8>>, index) -> !fir.ref<i8>
    fir.store %c1_i8 to %p : !fir.ref<i8>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @whole_block_accessed
// CHECK: acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.array<5000xi8>>) dataClause(acc_copy) implicit(true) name("com_")
