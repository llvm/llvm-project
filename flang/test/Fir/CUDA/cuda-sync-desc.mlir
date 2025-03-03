// RUN: fir-opt --cuf-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f64 = dense<64> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 20.0.0 (git@github.com:clementval/llvm-project.git f37e52237791f58438790c77edeb8de08f692987)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  fir.global @_QMdevptrEdev_ptr {data_attr = #cuf.cuda<device>} : !fir.box<!fir.ptr<!fir.array<?xf32>>> {
    %0 = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
    %c0 = arith.constant 0 : index
    %1 = fir.shape %c0 : (index) -> !fir.shape<1>
    %2 = fir.embox %0(%1) {allocator_idx = 2 : i32} : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
    fir.has_value %2 : !fir.box<!fir.ptr<!fir.array<?xf32>>>
  }
  func.func @_QQmain() {
    cuf.sync_descriptor @_QMdevptrEdev_ptr
    return
  }
}

// CHECK-LABEL: func.func @_QQmain()
// CHECK: %[[HOST_ADDR:.*]] = fir.address_of(@_QMdevptrEdev_ptr) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
// CHECK: %[[HOST_ADDR_PTR:.*]] = fir.convert %[[HOST_ADDR]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFSyncGlobalDescriptor(%[[HOST_ADDR_PTR]], %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.ref<i8>, i32)
