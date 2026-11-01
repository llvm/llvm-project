// RUN: fir-opt --cuf-convert %s | FileCheck %s

// Transfer from a dynamic-length character array into a descriptor. The
// descriptor built for the source must carry the length parameter.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QPs(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,64>>>>> {cuf.data_attr = #cuf.cuda<managed>, fir.bindc_name = "p"}, %n: index, %l: i32) {
    %c64 = arith.constant 64 : index
    %p = fir.declare %arg0 typeparams %c64 uniq_name("_QFsEp") fortran_attrs<pointer> data_attr(#cuf.cuda<managed>) : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,64>>>>>, index) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,64>>>>>
    %mem = fir.allocmem !fir.array<?x!fir.char<1,?>>(%l : i32), %n {uniq_name = "_QFsEstr"}
    %ref = fir.convert %mem : (!fir.heap<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
    %shape = fir.shape %n : (index) -> !fir.shape<1>
    %str = fir.declare %ref(%shape) typeparams %l uniq_name("_QFsEstr") : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
    cuf.data_transfer %str to %p, %shape : !fir.shape<1> transfer_kind(host_device) : !fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,64>>>>>
    fir.freemem %mem : !fir.heap<!fir.array<?x!fir.char<1,?>>>
    return
  }
}

// CHECK-LABEL: func.func @_QPs(
// CHECK-SAME:    %{{.*}}: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,64>>>>> {{.*}}, %{{.*}}: index, %[[L:.*]]: i32)
// CHECK:         %[[STR:.*]] = fir.declare %{{.*}}(%[[SHAPE:.*]]) typeparams %[[L]]
// CHECK:         fir.embox %[[STR]](%[[SHAPE]]) typeparams %[[L]] : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:         fir.call @_FortranACUFDataTransferDescDesc
