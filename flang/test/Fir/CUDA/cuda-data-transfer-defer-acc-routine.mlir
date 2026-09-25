// RUN: fir-opt --cuf-convert="defer-acc-routine-data-transfers=true" %s | FileCheck %s --check-prefix=DEFER
// RUN: fir-opt --cuf-convert %s | FileCheck %s --check-prefix=CONVERT

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  func.func @host(%dst: !fir.ref<i32>) {
    %c1_i32 = arith.constant 1 : i32
    cuf.data_transfer %c1_i32 to %dst {transfer_kind = #cuf.cuda_transfer<host_device>} : i32, !fir.ref<i32>
    return
  }

  func.func @acc_routine(%dst: !fir.ref<i32>) attributes {acc.routine_info = #acc.routine_info<[@routine]>} {
    %c1_i32 = arith.constant 1 : i32
    cuf.data_transfer %c1_i32 to %dst {transfer_kind = #cuf.cuda_transfer<host_device>} : i32, !fir.ref<i32>
    return
  }

  func.func @device_specialized() attributes {acc.specialized_routine = #acc.specialized_routine<@routine, <seq>, "device_specialized">} {
    acc.compute_region {
      %dst = fir.alloca i32
      %c1_i32 = arith.constant 1 : i32
      cuf.data_transfer %c1_i32 to %dst {transfer_kind = #cuf.cuda_transfer<host_device>} : i32, !fir.ref<i32>
      acc.yield
    } <{origin = "acc.routine"}>
    return
  }

  func.func @device_specialized_logical() attributes {acc.specialized_routine = #acc.specialized_routine<@routine, <seq>, "device_specialized_logical">} {
    acc.compute_region {
      %dst = fir.alloca !fir.logical<4>
      %true = arith.constant true
      cuf.data_transfer %true to %dst {transfer_kind = #cuf.cuda_transfer<host_device>} : i1, !fir.ref<!fir.logical<4>>
      acc.yield
    } <{origin = "acc.routine"}>
    return
  }

  // dst = src for explicit-shape arrays (host src, device dst) in an OpenACC
  // routine. The host copy keeps a transfer; the specialized device body
  // becomes an assignment.
  func.func @acc_routine_copy_array() attributes {acc.routine_info = #acc.routine_info<[@routine]>} {
    %n = arith.constant 10 : index
    %shape = fir.shape %n : (index) -> !fir.shape<1>
    %src = fir.alloca !fir.array<?xf32>, %n
    %dst = fir.alloca !fir.array<?xf32>, %n
    cuf.data_transfer %src to %dst, %shape : !fir.shape<1> {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.ref<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>
    return
  }

  func.func @device_specialized_copy_array() attributes {acc.specialized_routine = #acc.specialized_routine<@routine, <seq>, "device_specialized_copy_array">} {
    acc.compute_region {
      %n = arith.constant 10 : index
      %shape = fir.shape %n : (index) -> !fir.shape<1>
      %src = fir.alloca !fir.array<?xf32>, %n
      %dst = fir.alloca !fir.array<?xf32>, %n
      cuf.data_transfer %src to %dst, %shape : !fir.shape<1> {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.ref<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>
      acc.yield
    } <{origin = "acc.routine"}>
    return
  }
}

// DEFER-LABEL: func.func @host(
// DEFER-NOT: cuf.data_transfer
// DEFER: fir.call @_FortranACUFDataTransferPtrPtr

// DEFER-LABEL: func.func @acc_routine(
// DEFER: cuf.data_transfer
// DEFER-NOT: fir.call @_FortranACUFDataTransferPtrPtr

// DEFER-LABEL: func.func @acc_routine_copy_array(
// DEFER: cuf.data_transfer
// DEFER-NOT: fir.call @_FortranACUFDataTransferPtrPtr

// CONVERT-LABEL: func.func @host(
// CONVERT-NOT: cuf.data_transfer
// CONVERT: fir.call @_FortranACUFDataTransferPtrPtr

// CONVERT-LABEL: func.func @acc_routine(
// CONVERT-NOT: cuf.data_transfer
// CONVERT: fir.call @_FortranACUFDataTransferPtrPtr

// CONVERT-LABEL: func.func @device_specialized(
// CONVERT-NOT: cuf.data_transfer
// CONVERT-NOT: fir.call @_FortranACUFDataTransferPtrPtr
// CONVERT: fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>

// CONVERT-LABEL: func.func @device_specialized_logical(
// CONVERT-NOT: cuf.data_transfer
// CONVERT: %[[CVT:.*]] = fir.convert %{{.*}} : (i1) -> !fir.logical<4>
// CONVERT: fir.store %[[CVT]] to %{{.*}} : !fir.ref<!fir.logical<4>>

// CONVERT-LABEL: func.func @acc_routine_copy_array(
// CONVERT-NOT: cuf.data_transfer
// CONVERT: fir.call @_FortranACUFDataTransferPtrPtr

// CONVERT-LABEL: func.func @device_specialized_copy_array(
// CONVERT-NOT: cuf.data_transfer
// CONVERT-NOT: fir.call @_FortranACUFDataTransferPtrPtr
// CONVERT: %[[SRC:.*]] = fir.embox %{{.*}}(%{{.*}}) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
// CONVERT: %[[DST:.*]] = fir.embox %{{.*}}(%{{.*}}) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
// CONVERT: hlfir.assign %[[SRC]] to %[[DST]] : !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>
