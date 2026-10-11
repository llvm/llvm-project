// Under -gpu=mem:managed, runtime type information is shared between host and
// device so a descriptor in managed memory holds a type descriptor address
// valid on both sides.

// RUN: fir-opt --cuf-shared-type-info %s | FileCheck %s
// RUN: fir-opt --cuf-shared-type-info %s | FileCheck %s --check-prefix=NODECLARE
// RUN: fir-opt --cuf-shared-type-info --cuf-add-constructor="cuda-managed-type-info=true" %s | FileCheck %s --check-prefix=CTOR

module attributes {dlti.dl_spec = #dlti.dl_spec<i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, f80 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, "dlti.stack_alignment" = 128 : i64, "dlti.endianness" = "little">, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  fir.global linkonce_odr @_QMmE.n.t {acc.declare = #acc.declare<dataClause = acc_copyin>} constant : !fir.char<1> {
    %0 = fir.string_lit "t"(1) : !fir.char<1>
    fir.has_value %0 : !fir.char<1>
  }
  fir.global linkonce_odr @_QMmE.dt.t {acc.declare = #acc.declare<dataClause = acc_copyin>} constant : !fir.llvm_ptr<i8> {
    %0 = fir.address_of(@_QMmE.n.t) : !fir.ref<!fir.char<1>>
    %1 = fir.convert %0 : (!fir.ref<!fir.char<1>>) -> !fir.llvm_ptr<i8>
    fir.has_value %1 : !fir.llvm_ptr<i8>
  }
  // Type information defined in another unit, not used by device code.
  fir.global @_QMotherE.dt.u {acc.declare = #acc.declare<dataClause = acc_copyin>} constant : !fir.char<1>
  // Not type information.
  fir.global linkonce @_QQclX74 constant : !fir.char<1> {
    %0 = fir.string_lit "t"(1) : !fir.char<1>
    fir.has_value %0 : !fir.char<1>
  }
  fir.global @_QMmEx : i32 {
    %0 = arith.constant 0 : i32
    fir.has_value %0 : i32
  }

  gpu.module @cuda_device_mod {
    fir.global linkonce_odr @_QMmE.n.t {acc.declare = #acc.declare<dataClause = acc_copyin>} constant : !fir.char<1> {
      %0 = fir.string_lit "t"(1) : !fir.char<1>
      fir.has_value %0 : !fir.char<1>
    }
    fir.global linkonce_odr @_QMmE.dt.t {acc.declare = #acc.declare<dataClause = acc_copyin>} constant : !fir.llvm_ptr<i8> {
      %0 = fir.address_of(@_QMmE.n.t) : !fir.ref<!fir.char<1>>
      %1 = fir.convert %0 : (!fir.ref<!fir.char<1>>) -> !fir.llvm_ptr<i8>
      fir.has_value %1 : !fir.llvm_ptr<i8>
    }
    gpu.func @_QMmPk() kernel {
      gpu.return
    }
  }
}

// Host type information is writable and placed in the shared section. Each
// type descriptor used on the device gets a managed pointer.
// CHECK: fir.global linkonce_odr @_QMmE.n.t <{section = "__nv_type_info"}> : !fir.char<1> {
// CHECK: fir.global linkonce_odr @_QMmE.dt.t <{section = "__nv_type_info"}> : !fir.llvm_ptr<i8> {
// CHECK: fir.global @_QMmEXdtXtXhostaddr[[TAG:[0-9a-f]+]] <{data_attr = #cuf.cuda<managed>}> {cuf.host_type_desc = @_QMmE.dt.t} : !fir.llvm_ptr<i8> {
// CHECK:   fir.zero_bits !fir.llvm_ptr<i8>
// CHECK: fir.global @_QMotherE.dt.u constant : !fir.char<1>{{$}}
// CHECK: fir.global linkonce @_QQclX74 constant : !fir.char<1> {
// CHECK: fir.global @_QMmEx : i32 {

// The device copies stay, and the GPU module maps each shared type descriptor
// to its pointer.
// CHECK: gpu.module @cuda_device_mod
// CHECK-SAME: cuf.shared_type_descs
// CHECK-SAME: _QMmEXdtXt = @_QMmEXdtXtXhostaddr[[TAG]]
// CHECK-DAG: fir.global linkonce_odr @_QMmE.n.t constant : !fir.char<1> {
// CHECK-DAG: fir.global linkonce_odr @_QMmE.dt.t constant : !fir.llvm_ptr<i8> {
// CHECK-DAG: fir.global @_QMmEXdtXtXhostaddr[[TAG]] <{data_attr = #cuf.cuda<managed>}> : !fir.llvm_ptr<i8> {

// NODECLARE-NOT: acc.declare

// The constructor registers the pages of the section and the managed pointer,
// and saves the module handle. A second constructor, which runs after the
// registration constructors of all the units, initializes the module and then
// stores the host address of the type descriptor in the managed pointer,
// unless the runtime left the pointer null.
// CTOR-LABEL: llvm.func internal @__cudaFortranConstructor()
// CTOR: %[[START:.*]] = llvm.mlir.addressof @__start___nv_type_info : !llvm.ptr
// CTOR: %[[STOP:.*]] = llvm.mlir.addressof @__stop___nv_type_info : !llvm.ptr
// CTOR: llvm.call @_FortranACUFRegisterHostMemoryRange(%[[START]], %[[STOP]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CTOR: %[[MOD:.*]] = cuf.register_module @cuda_device_mod -> !llvm.ptr
// CTOR: fir.call @_FortranACUFRegisterManagedVariable
// CTOR-NOT: fir.call @_FortranACUFInitModule
// CTOR: %[[HANDLEADDR:.*]] = llvm.mlir.addressof @__cudaFortranModuleHandle : !llvm.ptr
// CTOR: llvm.store %[[MOD]], %[[HANDLEADDR]] : !llvm.ptr, !llvm.ptr
// CTOR: llvm.return
// CTOR-DAG: llvm.mlir.global extern_weak hidden @__start___nv_type_info() {{.*}}: i8
// CTOR-DAG: llvm.mlir.global extern_weak hidden @__stop___nv_type_info() {{.*}}: i8
// CTOR-DAG: llvm.mlir.global internal @__cudaFortranModuleHandle() {{.*}}: !llvm.ptr

// CTOR-LABEL: llvm.func internal @__cudaFortranInitConstructor()
// CTOR: %[[HANDLEADDR2:.*]] = llvm.mlir.addressof @__cudaFortranModuleHandle : !llvm.ptr
// CTOR: %[[HANDLE:.*]] = llvm.load %[[HANDLEADDR2]] : !llvm.ptr -> !llvm.ptr
// CTOR: fir.convert %[[HANDLE]]
// CTOR: fir.call @_FortranACUFInitModule
// CTOR: %[[PTRREF:.*]] = fir.address_of(@_QMmEXdtXtXhostaddr{{[0-9a-f]+}}.managed.ptr) : !fir.ref<!fir.llvm_ptr<i8>>
// CTOR: %[[MANAGED:.*]] = fir.load %[[PTRREF]] : !fir.ref<!fir.llvm_ptr<i8>>
// CTOR: %[[NOTNULL:.*]] = arith.cmpi ne
// CTOR: llvm.cond_br %[[NOTNULL]], ^[[STORE:bb[0-9]+]], ^[[NEXT:bb[0-9]+]]
// CTOR: ^[[STORE]]:
// CTOR: %[[DEST:.*]] = fir.convert %[[MANAGED]] : (!fir.llvm_ptr<i8>) -> !fir.ref<!fir.llvm_ptr<i8>>
// CTOR: %[[TYPEDESC:.*]] = fir.address_of(@_QMmE.dt.t) : !fir.ref<!fir.llvm_ptr<i8>>
// CTOR: %[[VAL:.*]] = fir.convert %[[TYPEDESC]] : (!fir.ref<!fir.llvm_ptr<i8>>) -> !fir.llvm_ptr<i8>
// CTOR: fir.store %[[VAL]] to %[[DEST]] : !fir.ref<!fir.llvm_ptr<i8>>
// CTOR: llvm.br ^[[NEXT]]
// CTOR: ^[[NEXT]]:
// CTOR: llvm.return

// CTOR: llvm.mlir.global_ctors ctors = [@__cudaFortranConstructor, @__cudaFortranInitConstructor], priorities = [0 : i32, 1 : i32]
