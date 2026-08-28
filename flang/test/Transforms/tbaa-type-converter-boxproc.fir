// Tests that the type converter handles record and sequence types containing
// !fir.boxproc members by returning std::nullopt instead of crashing.

// RUN: fir-opt --split-input-file --fir-add-alias-tags %s | FileCheck %s

// Record type containing a !fir.boxproc member in COMMON storage.
// convertRecordType returns std::nullopt for such types.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
  fir.global common @g1_(dense<0> : vector<16xi8>) {alignment = 8 : i64} : !fir.array<16xi8>
  func.func @_QPtest_record() {
    %c0 = arith.constant 0 : index
    %addr = fir.address_of(@g1_) : !fir.ref<!fir.array<16xi8>>
    %0 = fir.coordinate_of %addr, %c0 : (!fir.ref<!fir.array<16xi8>>, index) -> !fir.ref<i8>
    %1 = fir.convert %0 : (!fir.ref<i8>) -> !fir.ref<!fir.type<t_rec{proc:!fir.boxproc<()->()>}>>
    %2 = fir.declare %1 storage(%addr[0]) {uniq_name = "_QPtest_recordEv"} : (!fir.ref<!fir.type<t_rec{proc:!fir.boxproc<()->()>}>>, !fir.ref<!fir.array<16xi8>>) -> !fir.ref<!fir.type<t_rec{proc:!fir.boxproc<()->()>}>>
    %3 = fir.load %2 : !fir.ref<!fir.type<t_rec{proc:!fir.boxproc<()->()>}>>
    fir.store %3 to %2 : !fir.ref<!fir.type<t_rec{proc:!fir.boxproc<()->()>}>>
    return
  }
}
// No byte-range TBAA should be generated.
// CHECK-NOT: bytes_
// CHECK-LABEL: func.func @_QPtest_record
// CHECK: return

// -----

// Array of records containing a !fir.boxproc member in COMMON storage.
// convertSequenceType returns std::nullopt for such types.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
  fir.global common @g2_(dense<0> : vector<160xi8>) {alignment = 8 : i64} : !fir.array<160xi8>
  func.func @_QPtest_sequence() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %addr = fir.address_of(@g2_) : !fir.ref<!fir.array<160xi8>>
    %0 = fir.coordinate_of %addr, %c0 : (!fir.ref<!fir.array<160xi8>>, index) -> !fir.ref<i8>
    %1 = fir.convert %0 : (!fir.ref<i8>) -> !fir.ref<!fir.array<10x!fir.type<t_rec{proc:!fir.boxproc<()->()>}>>>
    %shape = fir.shape %c10 : (index) -> !fir.shape<1>
    %2 = fir.declare %1(%shape) storage(%addr[0]) {uniq_name = "_QPtest_sequenceEv"} : (!fir.ref<!fir.array<10x!fir.type<t_rec{proc:!fir.boxproc<()->()>}>>>, !fir.shape<1>, !fir.ref<!fir.array<160xi8>>) -> !fir.ref<!fir.array<10x!fir.type<t_rec{proc:!fir.boxproc<()->()>}>>>
    %3 = fir.array_coor %2(%shape) %c1 : (!fir.ref<!fir.array<10x!fir.type<t_rec{proc:!fir.boxproc<()->()>}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<t_rec{proc:!fir.boxproc<()->()>}>>
    %4 = fir.load %3 : !fir.ref<!fir.type<t_rec{proc:!fir.boxproc<()->()>}>>
    fir.store %4 to %3 : !fir.ref<!fir.type<t_rec{proc:!fir.boxproc<()->()>}>>
    return
  }
}
// No byte-range TBAA should be generated since convertSequenceType returns nullopt.
// CHECK-NOT: bytes_
// CHECK-LABEL: func.func @_QPtest_sequence
// CHECK: return
