// RUN: fir-opt --target-rewrite="target=i386-unknown-linux-gnu" %s | FileCheck %s --check-prefix=I32
// RUN: fir-opt --target-rewrite="target=x86_64-unknown-linux-gnu" %s | FileCheck %s --check-prefix=X64
// RUN: fir-opt --target-rewrite="target=aarch64-unknown-linux-gnu" %s | FileCheck %s --check-prefix=AARCH64
// RUN: fir-opt --target-rewrite="target=powerpc64le-unknown-linux-gnu" %s | FileCheck %s --check-prefix=PPC

// Test that we rewrite the signature and body of a function that returns a
// complex<4>.
// I32-LABEL: func @returncomplex4() -> i64
// X64-LABEL: func @returncomplex4() -> !fir.vector<2:!fir.real<4>>
// AARCH64-LABEL: func @returncomplex4() -> tuple<!fir.real<4>, !fir.real<4>>
// PPC-LABEL: func @returncomplex4() -> tuple<!fir.real<4>, !fir.real<4>>
func.func @returncomplex4() -> !fir.complex<4> {
  // I32: fir.insert_value
  // I32: [[VAL:%[0-9A-Za-z]+]] = fir.insert_value
  // X64: fir.insert_value
  // X64: [[VAL:%[0-9A-Za-z]+]] = fir.insert_value
  // AARCH64: fir.insert_value
  // AARCH64: [[VAL:%[0-9A-Za-z]+]] = fir.insert_value
  // PPC: fir.insert_value
  // PPC: [[VAL:%[0-9A-Za-z]+]] = fir.insert_value
  %1 = fir.undefined !fir.complex<4>
  %2 = arith.constant 2.0 : f32
  %3 = fir.convert %2 : (f32) -> !fir.real<4>
  %c0 = arith.constant 0 : i32
  %4 = fir.insert_value %1, %3, [0 : i32] : (!fir.complex<4>, !fir.real<4>) -> !fir.complex<4>
  %c1 = arith.constant 1 : i32
  %5 = arith.constant -42.0 : f32
  %6 = fir.insert_value %4, %5, [1 : i32] : (!fir.complex<4>, f32) -> !fir.complex<4>

  // I32: [[ADDRI64:%[0-9A-Za-z]+]] = fir.alloca i64
  // I32: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRI64]] : (!fir.ref<i64>) -> !fir.ref<!fir.complex<4>>
  // I32: fir.store [[VAL]] to [[ADDRC]] : !fir.ref<!fir.complex<4>>
  // I32: [[RES:%[0-9A-Za-z]+]] = fir.load [[ADDRI64]] : !fir.ref<i64>
  // I32: return [[RES]] : i64
  // X64: [[ADDRV:%[0-9A-Za-z]+]] = fir.alloca !fir.vector<2:!fir.real<4>>
  // X64: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRV]] : (!fir.ref<!fir.vector<2:!fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // X64: fir.store [[VAL]] to [[ADDRC]] : !fir.ref<!fir.complex<4>>
  // X64: [[RES:%[0-9A-Za-z]+]] = fir.load [[ADDRV]] : !fir.ref<!fir.vector<2:!fir.real<4>>>
  // X64: return [[RES]] : !fir.vector<2:!fir.real<4>>
  // AARCH64: [[ADDRT:%[0-9A-Za-z]+]] = fir.alloca tuple<!fir.real<4>, !fir.real<4>>
  // AARCH64: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRT]] : (!fir.ref<tuple<!fir.real<4>, !fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // AARCH64: fir.store [[VAL]] to [[ADDRC]] : !fir.ref<!fir.complex<4>>
  // AARCH64: [[RES:%[0-9A-Za-z]+]] = fir.load [[ADDRT]] : !fir.ref<tuple<!fir.real<4>, !fir.real<4>>>
  // AARCH64: return [[RES]] : tuple<!fir.real<4>, !fir.real<4>>
  // PPC: [[ADDRT:%[0-9A-Za-z]+]] = fir.alloca tuple<!fir.real<4>, !fir.real<4>>
  // PPC: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRT]] : (!fir.ref<tuple<!fir.real<4>, !fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // PPC: fir.store [[VAL]] to [[ADDRC]] : !fir.ref<!fir.complex<4>>
  // PPC: [[RES:%[0-9A-Za-z]+]] = fir.load [[ADDRT]] : !fir.ref<tuple<!fir.real<4>, !fir.real<4>>>
  // PPC: return [[RES]] : tuple<!fir.real<4>, !fir.real<4>>
  return %6 : !fir.complex<4>
}

// Test that we rewrite the signature and body of a function that returns a
// complex<8>.
// I32-LABEL:func @returncomplex8
// I32-SAME: ([[ARG0:%[0-9A-Za-z]+]]: !fir.ref<tuple<!fir.real<8>, !fir.real<8>>>  {llvm.align = 4 : i32, llvm.sret = tuple<!fir.real<8>, !fir.real<8>>})
// X64-LABEL: func @returncomplex8() -> tuple<!fir.real<8>, !fir.real<8>>
// AARCH64-LABEL: func @returncomplex8() -> tuple<!fir.real<8>, !fir.real<8>>
// PPC-LABEL: func @returncomplex8() -> tuple<!fir.real<8>, !fir.real<8>>
func.func @returncomplex8() -> !fir.complex<8> {
  // I32: fir.insert_value
  // I32: [[VAL:%[0-9A-Za-z]+]] = fir.insert_value {{.*}}
  // X64: fir.insert_value
  // X64: [[VAL:%[0-9A-Za-z]+]] = fir.insert_value {{.*}}
  // AARCH64: fir.insert_value
  // AARCH64: [[VAL:%[0-9A-Za-z]+]] = fir.insert_value {{.*}}
  // PPC: fir.insert_value
  // PPC: [[VAL:%[0-9A-Za-z]+]] = fir.insert_value {{.*}}
  %1 = fir.undefined !fir.complex<8>
  %2 = arith.constant 1.0 : f64
  %3 = arith.constant -4.0 : f64
  %c0 = arith.constant 0 : i32
  %4 = fir.insert_value %1, %3, [0 : i32] : (!fir.complex<8>, f64) -> !fir.complex<8>
  %c1 = arith.constant 1 : i32
  %5 = fir.insert_value %4, %2, [1 : i32] : (!fir.complex<8>, f64) -> !fir.complex<8>

  // I32: [[ADDR:%[0-9A-Za-z]+]] = fir.convert [[ARG0]] : (!fir.ref<tuple<!fir.real<8>, !fir.real<8>>>) -> !fir.ref<!fir.complex<8>>
  // I32: fir.store [[VAL]] to [[ADDR]] : !fir.ref<!fir.complex<8>>
  // I32: return{{ *$}}
  // X64: [[ADDRT:%[0-9A-Za-z]+]] = fir.alloca tuple<!fir.real<8>, !fir.real<8>>
  // X64: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRT]] : (!fir.ref<tuple<!fir.real<8>, !fir.real<8>>>) -> !fir.ref<!fir.complex<8>>
  // X64: fir.store [[VAL]] to [[ADDRC]] : !fir.ref<!fir.complex<8>>
  // X64: [[RES:%[0-9A-Za-z]+]] = fir.load [[ADDRT]] : !fir.ref<tuple<!fir.real<8>, !fir.real<8>>>
  // X64: return [[RES]] : tuple<!fir.real<8>, !fir.real<8>>
  // AARCH64: [[ADDRT:%[0-9A-Za-z]+]] = fir.alloca tuple<!fir.real<8>, !fir.real<8>>
  // AARCH64: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRT]] : (!fir.ref<tuple<!fir.real<8>, !fir.real<8>>>) -> !fir.ref<!fir.complex<8>>
  // AARCH64: fir.store [[VAL]] to [[ADDRC]] : !fir.ref<!fir.complex<8>>
  // AARCH64: [[RES:%[0-9A-Za-z]+]] = fir.load [[ADDRT]] : !fir.ref<tuple<!fir.real<8>, !fir.real<8>>>
  // AARCH64: return [[RES]] : tuple<!fir.real<8>, !fir.real<8>>
  // PPC: [[ADDRT:%[0-9A-Za-z]+]] = fir.alloca tuple<!fir.real<8>, !fir.real<8>>
  // PPC: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRT]] : (!fir.ref<tuple<!fir.real<8>, !fir.real<8>>>) -> !fir.ref<!fir.complex<8>>
  // PPC: fir.store [[VAL]] to [[ADDRC]] : !fir.ref<!fir.complex<8>>
  // PPC: [[RES:%[0-9A-Za-z]+]] = fir.load [[ADDRT]] : !fir.ref<tuple<!fir.real<8>, !fir.real<8>>>
  // PPC: return [[RES]] : tuple<!fir.real<8>, !fir.real<8>>
  return %5 : !fir.complex<8>
}

// Test that we rewrite the signature of a function that accepts a complex<4>.
// I32-LABEL: func private @paramcomplex4(!fir.ref<tuple<!fir.real<4>, !fir.real<4>>> {llvm.align = 4 : i32, llvm.byval = tuple<!fir.real<4>, !fir.real<4>>})
// X64-LABEL: func private @paramcomplex4(!fir.vector<2:!fir.real<4>>)
// AARCH64-LABEL: func private @paramcomplex4(!fir.array<2x!fir.real<4>>)
// PPC-LABEL: func private @paramcomplex4(!fir.real<4>, !fir.real<4>)
func.func private @paramcomplex4(!fir.complex<4>) -> ()

// Test that we rewrite calls to functions that return or accept complex<4>.
// I32-LABEL: func @callcomplex4()
// X64-LABEL: func @callcomplex4()
// AARCH64-LABEL: func @callcomplex4()
// PPC-LABEL: func @callcomplex4()
func.func @callcomplex4() {

  // I32: [[RES:%[0-9A-Za-z]+]] = fir.call @returncomplex4() : () -> i64
  // X64: [[RES:%[0-9A-Za-z]+]] = fir.call @returncomplex4() : () -> !fir.vector<2:!fir.real<4>>
  // AARCH64: [[RES:%[0-9A-Za-z]+]] = fir.call @returncomplex4() : () -> tuple<!fir.real<4>, !fir.real<4>>
  // PPC: [[RES:%[0-9A-Za-z]+]] = fir.call @returncomplex4() : () -> tuple<!fir.real<4>, !fir.real<4>>
  %1 = fir.call @returncomplex4() : () -> !fir.complex<4>

  // I32: [[ADDRI64:%[0-9A-Za-z]+]] = fir.alloca i64
  // I32: fir.store [[RES]] to [[ADDRI64]] : !fir.ref<i64>
  // I32: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRI64]] : (!fir.ref<i64>) -> !fir.ref<!fir.complex<4>>
  // I32: [[C:%[0-9A-Za-z]+]] = fir.load [[ADDRC]] : !fir.ref<!fir.complex<4>>
  // I32: [[ADDRC2:%[0-9A-Za-z]+]] = fir.alloca !fir.complex<4>
  // I32: fir.store [[C]] to [[ADDRC2]] : !fir.ref<!fir.complex<4>>
  // I32: [[T:%[0-9A-Za-z]+]] = fir.convert [[ADDRC2]] : (!fir.ref<!fir.complex<4>>) -> !fir.ref<tuple<!fir.real<4>, !fir.real<4>>>
  // I32: fir.call @paramcomplex4([[T]]) : (!fir.ref<tuple<!fir.real<4>, !fir.real<4>>>) -> ()

  // X64: [[ADDRV:%[0-9A-Za-z]+]] = fir.alloca !fir.vector<2:!fir.real<4>>
  // X64: fir.store [[RES]] to [[ADDRV]] : !fir.ref<!fir.vector<2:!fir.real<4>>>
  // X64: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRV]] : (!fir.ref<!fir.vector<2:!fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // X64: [[V:%[0-9A-Za-z]+]] = fir.load [[ADDRC]] : !fir.ref<!fir.complex<4>>
  // X64: [[ADDRV2:%[0-9A-Za-z]+]] = fir.alloca !fir.vector<2:!fir.real<4>>
  // X64: [[ADDRC2:%[0-9A-Za-z]+]] = fir.convert [[ADDRV2]] : (!fir.ref<!fir.vector<2:!fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // X64: fir.store [[V]] to [[ADDRC2]] : !fir.ref<!fir.complex<4>>
  // X64: [[VRELOADED:%[0-9A-Za-z]+]] = fir.load [[ADDRV2]] : !fir.ref<!fir.vector<2:!fir.real<4>>>
  // X64: fir.call @paramcomplex4([[VRELOADED]]) : (!fir.vector<2:!fir.real<4>>) -> ()

  // AARCH64: [[ADDRT:%[0-9A-Za-z]+]] = fir.alloca tuple<!fir.real<4>, !fir.real<4>>
  // AARCH64: fir.store [[RES]] to [[ADDRT]] : !fir.ref<tuple<!fir.real<4>, !fir.real<4>>>
  // AARCH64: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRT]] : (!fir.ref<tuple<!fir.real<4>, !fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // AARCH64: [[V:%[0-9A-Za-z]+]] = fir.load [[ADDRC]] : !fir.ref<!fir.complex<4>>
  // AARCH64: [[ADDRARR:%[0-9A-Za-z]+]] = fir.alloca !fir.array<2x!fir.real<4>>
  // AARCH64: [[ADDRC2:%[0-9A-Za-z]+]] = fir.convert [[ADDRARR]] : (!fir.ref<!fir.array<2x!fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // AARCH64: fir.store [[V]] to [[ADDRC2]] : !fir.ref<!fir.complex<4>>
  // AARCH64: [[ARR:%[0-9A-Za-z]+]] = fir.load [[ADDRARR]] : !fir.ref<!fir.array<2x!fir.real<4>>>
  // AARCH64: fir.call @paramcomplex4([[ARR]]) : (!fir.array<2x!fir.real<4>>) -> ()

  // PPC: [[ADDRT:%[0-9A-Za-z]+]] = fir.alloca tuple<!fir.real<4>, !fir.real<4>>
  // PPC: fir.store [[RES]] to [[ADDRT]] : !fir.ref<tuple<!fir.real<4>, !fir.real<4>>>
  // PPC: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRT]] : (!fir.ref<tuple<!fir.real<4>, !fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // PPC: [[V:%[0-9A-Za-z]+]] = fir.load [[ADDRC]] : !fir.ref<!fir.complex<4>>
  // PPC: [[A:%[0-9A-Za-z]+]] = fir.extract_value [[V]], [0 : i32] : (!fir.complex<4>) -> !fir.real<4>
  // PPC: [[B:%[0-9A-Za-z]+]] = fir.extract_value [[V]], [1 : i32] : (!fir.complex<4>) -> !fir.real<4>
  // PPC: fir.call @paramcomplex4([[A]], [[B]]) : (!fir.real<4>, !fir.real<4>) -> ()
  fir.call @paramcomplex4(%1) : (!fir.complex<4>) -> ()
  return
}

// Test that we rewrite the signature of a function that accepts a complex<8>.
// I32-LABEL: func private @paramcomplex8(!fir.ref<tuple<!fir.real<8>, !fir.real<8>>> {llvm.align = 4 : i32, llvm.byval = tuple<!fir.real<8>, !fir.real<8>>})
// X64-LABEL: func private @paramcomplex8(!fir.real<8>, !fir.real<8>)
// AARCH64-LABEL: func private @paramcomplex8(!fir.array<2x!fir.real<8>>)
// PPC-LABEL: func private @paramcomplex8(!fir.real<8>, !fir.real<8>)
func.func private @paramcomplex8(!fir.complex<8>) -> ()

// Test that we rewrite calls to functions that return or accept complex<8>.
// I32-LABEL: func @callcomplex8()
// X64-LABEL: func @callcomplex8()
// AARCH64-LABEL: func @callcomplex8()
// PPC-LABEL: func @callcomplex8()
func.func @callcomplex8() {
  // I32: [[RES:%[0-9A-Za-z]+]] = fir.alloca tuple<!fir.real<8>, !fir.real<8>>
  // I32: fir.call @returncomplex8([[RES]]) : (!fir.ref<tuple<!fir.real<8>, !fir.real<8>>>) -> ()
  // X64: [[RES:%[0-9A-Za-z]+]] = fir.call @returncomplex8() : () -> tuple<!fir.real<8>, !fir.real<8>>
  // AARCH64: [[RES:%[0-9A-Za-z]+]] = fir.call @returncomplex8() : () -> tuple<!fir.real<8>, !fir.real<8>>
  // PPC: [[RES:%[0-9A-Za-z]+]] = fir.call @returncomplex8() : () -> tuple<!fir.real<8>, !fir.real<8>>
  %1 = fir.call @returncomplex8() : () -> !fir.complex<8>

  // I32: [[RESC:%[0-9A-Za-z]+]] = fir.convert [[RES]] : (!fir.ref<tuple<!fir.real<8>, !fir.real<8>>>) -> !fir.ref<!fir.complex<8>>
  // I32: [[V:%[0-9A-Za-z]+]]  = fir.load [[RESC]] : !fir.ref<!fir.complex<8>>
  // I32: [[ADDRC:%[0-9A-Za-z]+]] = fir.alloca !fir.complex<8>
  // I32: fir.store [[V]] to [[ADDRC]] : !fir.ref<!fir.complex<8>>
  // I32: [[ADDRT:%[0-9A-Za-z]+]] = fir.convert [[ADDRC]] : (!fir.ref<!fir.complex<8>>) -> !fir.ref<tuple<!fir.real<8>, !fir.real<8>>>
  // I32: fir.call @paramcomplex8([[ADDRT]]) : (!fir.ref<tuple<!fir.real<8>, !fir.real<8>>>) -> ()

  // X64: [[ADDRT:%[0-9A-Za-z]+]] = fir.alloca tuple<!fir.real<8>, !fir.real<8>>
  // X64: fir.store [[RES]] to [[ADDRT]] : !fir.ref<tuple<!fir.real<8>, !fir.real<8>>>
  // X64: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRT]] : (!fir.ref<tuple<!fir.real<8>, !fir.real<8>>>) -> !fir.ref<!fir.complex<8>>
  // X64: [[V:%[0-9A-Za-z]+]] = fir.load [[ADDRC]] : !fir.ref<!fir.complex<8>>
  // X64: [[A:%[0-9A-Za-z]+]] = fir.extract_value [[V]], [0 : i32] : (!fir.complex<8>) -> !fir.real<8>
  // X64: [[B:%[0-9A-Za-z]+]] = fir.extract_value [[V]], [1 : i32] : (!fir.complex<8>) -> !fir.real<8>
  // X64: fir.call @paramcomplex8([[A]], [[B]]) : (!fir.real<8>, !fir.real<8>) -> ()

  // AARCH64: [[ADDRT:%[0-9A-Za-z]+]] = fir.alloca tuple<!fir.real<8>, !fir.real<8>>
  // AARCH64: fir.store [[RES]] to [[ADDRT]] : !fir.ref<tuple<!fir.real<8>, !fir.real<8>>>
  // AARCH64: [[ADDRV:%[0-9A-Za-z]+]] = fir.convert [[ADDRT]] : (!fir.ref<tuple<!fir.real<8>, !fir.real<8>>>) -> !fir.ref<!fir.complex<8>>
  // AARCH64: [[V:%[0-9A-Za-z]+]] = fir.load [[ADDRV]] : !fir.ref<!fir.complex<8>>
  // AARCH64: [[ADDRARR:%[0-9A-Za-z]+]] = fir.alloca !fir.array<2x!fir.real<8>>
  // AARCH64: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRARR]] : (!fir.ref<!fir.array<2x!fir.real<8>>>) -> !fir.ref<!fir.complex<8>>
  // AARCH64: fir.store [[V]] to [[ADDRC]] : !fir.ref<!fir.complex<8>>
  // AARCH64: [[ARR:%[0-9A-Za-z]+]] = fir.load [[ADDRARR]] : !fir.ref<!fir.array<2x!fir.real<8>>>
  // AARCH64: fir.call @paramcomplex8([[ARR]]) : (!fir.array<2x!fir.real<8>>) -> ()

  // PPC: [[ADDRT:%[0-9A-Za-z]+]] = fir.alloca tuple<!fir.real<8>, !fir.real<8>>
  // PPC: fir.store [[RES]] to [[ADDRT]] : !fir.ref<tuple<!fir.real<8>, !fir.real<8>>>
  // PPC: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRT]] : (!fir.ref<tuple<!fir.real<8>, !fir.real<8>>>) -> !fir.ref<!fir.complex<8>>
  // PPC: [[V:%[0-9A-Za-z]+]] = fir.load [[ADDRC]] : !fir.ref<!fir.complex<8>>
  // PPC: [[A:%[0-9A-Za-z]+]] = fir.extract_value [[V]], [0 : i32] : (!fir.complex<8>) -> !fir.real<8>
  // PPC: [[B:%[0-9A-Za-z]+]] = fir.extract_value [[V]], [1 : i32] : (!fir.complex<8>) -> !fir.real<8>
  // PPC: fir.call @paramcomplex8([[A]], [[B]]) : (!fir.real<8>, !fir.real<8>) -> ()
  fir.call @paramcomplex8(%1) : (!fir.complex<8>) -> ()
  return
}

// Test multiple complex<4> parameters and arguments
// I32-LABEL: func private @calleemultipleparamscomplex4(!fir.ref<tuple<!fir.real<4>, !fir.real<4>>> {llvm.align = 4 : i32, llvm.byval = tuple<!fir.real<4>, !fir.real<4>>}, !fir.ref<tuple<!fir.real<4>, !fir.real<4>>> {llvm.align = 4 : i32, llvm.byval = tuple<!fir.real<4>, !fir.real<4>>}, !fir.ref<tuple<!fir.real<4>, !fir.real<4>>> {llvm.align = 4 : i32, llvm.byval = tuple<!fir.real<4>, !fir.real<4>>})
// X64-LABEL: func private @calleemultipleparamscomplex4(!fir.vector<2:!fir.real<4>>, !fir.vector<2:!fir.real<4>>, !fir.vector<2:!fir.real<4>>)
// AARCH64-LABEL: func private @calleemultipleparamscomplex4(!fir.array<2x!fir.real<4>>, !fir.array<2x!fir.real<4>>, !fir.array<2x!fir.real<4>>)
// PPC-LABEL: func private @calleemultipleparamscomplex4(!fir.real<4>, !fir.real<4>, !fir.real<4>, !fir.real<4>, !fir.real<4>, !fir.real<4>)
func.func private @calleemultipleparamscomplex4(!fir.complex<4>, !fir.complex<4>, !fir.complex<4>) -> ()

// I32-LABEL: func @multipleparamscomplex4
// I32-SAME: ([[Z1:%[0-9A-Za-z]+]]: !fir.ref<tuple<!fir.real<4>, !fir.real<4>>> {llvm.align = 4 : i32, llvm.byval = tuple<!fir.real<4>, !fir.real<4>>}, [[Z2:%[0-9A-Za-z]+]]: !fir.ref<tuple<!fir.real<4>, !fir.real<4>>> {llvm.align = 4 : i32, llvm.byval = tuple<!fir.real<4>, !fir.real<4>>}, [[Z3:%[0-9A-Za-z]+]]: !fir.ref<tuple<!fir.real<4>, !fir.real<4>>> {llvm.align = 4 : i32, llvm.byval = tuple<!fir.real<4>, !fir.real<4>>})
// X64-LABEL: func @multipleparamscomplex4
// X64-SAME: ([[Z1:%[0-9A-Za-z]+]]: !fir.vector<2:!fir.real<4>>, [[Z2:%[0-9A-Za-z]+]]: !fir.vector<2:!fir.real<4>>, [[Z3:%[0-9A-Za-z]+]]: !fir.vector<2:!fir.real<4>>)
// AARCH64-LABEL: func @multipleparamscomplex4
// AARCH64-SAME: ([[Z1:%[0-9A-Za-z]+]]: !fir.array<2x!fir.real<4>>, [[Z2:%[0-9A-Za-z]+]]: !fir.array<2x!fir.real<4>>, [[Z3:%[0-9A-Za-z]+]]: !fir.array<2x!fir.real<4>>)
// PPC-LABEL: func @multipleparamscomplex4
// PPC-SAME: ([[A1:%[0-9A-Za-z]+]]: !fir.real<4>, [[B1:%[0-9A-Za-z]+]]: !fir.real<4>, [[A2:%[0-9A-Za-z]+]]: !fir.real<4>, [[B2:%[0-9A-Za-z]+]]: !fir.real<4>, [[A3:%[0-9A-Za-z]+]]: !fir.real<4>, [[B3:%[0-9A-Za-z]+]]: !fir.real<4>)
func.func @multipleparamscomplex4(%z1 : !fir.complex<4>, %z2 : !fir.complex<4>, %z3 : !fir.complex<4>) {
  // I32-DAG: [[Z1_ADDR:%[0-9A-Za-z]+]] = fir.convert [[Z1]] : (!fir.ref<tuple<!fir.real<4>, !fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // I32-DAG: [[Z1_VAL:%[0-9A-Za-z]+]] = fir.load [[Z1_ADDR]] : !fir.ref<!fir.complex<4>>
  // I32-DAG: [[Z2_ADDR:%[0-9A-Za-z]+]] = fir.convert [[Z2]] : (!fir.ref<tuple<!fir.real<4>, !fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // I32-DAG: [[Z2_VAL:%[0-9A-Za-z]+]] = fir.load [[Z2_ADDR]] : !fir.ref<!fir.complex<4>>
  // I32-DAG: [[Z3_ADDR:%[0-9A-Za-z]+]] = fir.convert [[Z3]] : (!fir.ref<tuple<!fir.real<4>, !fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // I32-DAG: [[Z3_VAL:%[0-9A-Za-z]+]] = fir.load [[Z3_ADDR]] : !fir.ref<!fir.complex<4>>

  // I32-DAG: [[Z1_ADDRC:%[0-9A-Za-z]+]] = fir.alloca !fir.complex<4>
  // I32-DAG: fir.store [[Z1_VAL]] to [[Z1_ADDRC]] : !fir.ref<!fir.complex<4>>
  // I32-DAG: [[Z1_ADDRT:%[0-9A-Za-z]+]] = fir.convert [[Z1_ADDRC]] : (!fir.ref<!fir.complex<4>>) -> !fir.ref<tuple<!fir.real<4>, !fir.real<4>>>
  // I32-DAG: [[Z2_ADDRC:%[0-9A-Za-z]+]] = fir.alloca !fir.complex<4>
  // I32-DAG: fir.store [[Z2_VAL]] to [[Z2_ADDRC]] : !fir.ref<!fir.complex<4>>
  // I32-DAG: [[Z2_ADDRT:%[0-9A-Za-z]+]] = fir.convert [[Z2_ADDRC]] : (!fir.ref<!fir.complex<4>>) -> !fir.ref<tuple<!fir.real<4>, !fir.real<4>>>
  // I32-DAG: [[Z3_ADDRC:%[0-9A-Za-z]+]] = fir.alloca !fir.complex<4>
  // I32-DAG: fir.store [[Z3_VAL]] to [[Z3_ADDRC]] : !fir.ref<!fir.complex<4>>
  // I32-DAG: [[Z3_ADDRT:%[0-9A-Za-z]+]] = fir.convert [[Z3_ADDRC]] : (!fir.ref<!fir.complex<4>>) -> !fir.ref<tuple<!fir.real<4>, !fir.real<4>>>

  // I32: fir.call @calleemultipleparamscomplex4([[Z1_ADDRT]], [[Z2_ADDRT]], [[Z3_ADDRT]]) : (!fir.ref<tuple<!fir.real<4>, !fir.real<4>>>, !fir.ref<tuple<!fir.real<4>, !fir.real<4>>>, !fir.ref<tuple<!fir.real<4>, !fir.real<4>>>) -> ()

  // X64-DAG: [[Z3_ADDR:%[0-9A-Za-z]+]] = fir.alloca !fir.vector<2:!fir.real<4>>
  // X64-DAG: fir.store [[Z3]] to [[Z3_ADDR]] : !fir.ref<!fir.vector<2:!fir.real<4>>>
  // X64-DAG: [[Z3_ADDRC:%[0-9A-Za-z]+]] = fir.convert [[Z3_ADDR]] : (!fir.ref<!fir.vector<2:!fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // X64-DAG: [[Z3_VAL:%[0-9A-Za-z]+]] = fir.load [[Z3_ADDRC]] : !fir.ref<!fir.complex<4>>
  // X64-DAG: [[Z2_ADDR:%[0-9A-Za-z]+]] = fir.alloca !fir.vector<2:!fir.real<4>>
  // X64-DAG: fir.store [[Z2]] to [[Z2_ADDR]] : !fir.ref<!fir.vector<2:!fir.real<4>>>
  // X64-DAG: [[Z2_ADDRC:%[0-9A-Za-z]+]] = fir.convert [[Z2_ADDR]] : (!fir.ref<!fir.vector<2:!fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // X64-DAG: [[Z2_VAL:%[0-9A-Za-z]+]] = fir.load [[Z2_ADDRC]] : !fir.ref<!fir.complex<4>>
  // X64-DAG: [[Z1_ADDR:%[0-9A-Za-z]+]] = fir.alloca !fir.vector<2:!fir.real<4>>
  // X64-DAG: fir.store [[Z1]] to [[Z1_ADDR]] : !fir.ref<!fir.vector<2:!fir.real<4>>>
  // X64-DAG: [[Z1_ADDRC:%[0-9A-Za-z]+]] = fir.convert [[Z1_ADDR]] : (!fir.ref<!fir.vector<2:!fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // X64-DAG: [[Z1_VAL:%[0-9A-Za-z]+]] = fir.load [[Z1_ADDRC]] : !fir.ref<!fir.complex<4>>

  // X64-DAG: [[Z1_ADDRV:%[0-9A-Za-z]+]] = fir.alloca !fir.vector<2:!fir.real<4>>
  // X64-DAG: [[Z1_ADDRC2:%[0-9A-Za-z]+]] = fir.convert [[Z1_ADDRV]] : (!fir.ref<!fir.vector<2:!fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // X64-DAG: fir.store [[Z1_VAL]] to [[Z1_ADDRC2]] : !fir.ref<!fir.complex<4>>
  // X64-DAG: [[Z1_RELOADED:%[0-9A-Za-z]+]] = fir.load [[Z1_ADDRV]] : !fir.ref<!fir.vector<2:!fir.real<4>>>
  // X64-DAG: [[Z2_ADDRV:%[0-9A-Za-z]+]] = fir.alloca !fir.vector<2:!fir.real<4>>
  // X64-DAG: [[Z2_ADDRC2:%[0-9A-Za-z]+]] = fir.convert [[Z2_ADDRV]] : (!fir.ref<!fir.vector<2:!fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // X64-DAG: fir.store [[Z2_VAL]] to [[Z2_ADDRC2]] : !fir.ref<!fir.complex<4>>
  // X64-DAG: [[Z2_RELOADED:%[0-9A-Za-z]+]] = fir.load [[Z2_ADDRV]] : !fir.ref<!fir.vector<2:!fir.real<4>>>
  // X64-DAG: [[Z3_ADDRV:%[0-9A-Za-z]+]] = fir.alloca !fir.vector<2:!fir.real<4>>
  // X64-DAG: [[Z3_ADDRC2:%[0-9A-Za-z]+]] = fir.convert [[Z3_ADDRV]] : (!fir.ref<!fir.vector<2:!fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // X64-DAG: fir.store [[Z3_VAL]] to [[Z3_ADDRC2]] : !fir.ref<!fir.complex<4>>
  // X64-DAG: [[Z3_RELOADED:%[0-9A-Za-z]+]] = fir.load [[Z3_ADDRV]] : !fir.ref<!fir.vector<2:!fir.real<4>>>

  // X64: fir.call @calleemultipleparamscomplex4([[Z1_RELOADED]], [[Z2_RELOADED]], [[Z3_RELOADED]]) : (!fir.vector<2:!fir.real<4>>, !fir.vector<2:!fir.real<4>>, !fir.vector<2:!fir.real<4>>) -> ()

  // AARCH64-DAG: [[Z3_ARR:%[0-9A-Za-z]+]] = fir.alloca !fir.array<2x!fir.real<4>>
  // AARCH64-DAG: fir.store [[Z3]] to [[Z3_ARR]] : !fir.ref<!fir.array<2x!fir.real<4>>>
  // AARCH64-DAG: [[Z3_ADDRC:%[0-9A-Za-z]+]] = fir.convert [[Z3_ARR]] : (!fir.ref<!fir.array<2x!fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // AARCH64-DAG: [[Z3_VAL:%[0-9A-Za-z]+]] = fir.load [[Z3_ADDRC]] : !fir.ref<!fir.complex<4>>
  // AARCH64-DAG: [[Z2_ARR:%[0-9A-Za-z]+]] = fir.alloca !fir.array<2x!fir.real<4>>
  // AARCH64-DAG: fir.store [[Z2]] to [[Z2_ARR]] : !fir.ref<!fir.array<2x!fir.real<4>>>
  // AARCH64-DAG: [[Z2_ADDRC:%[0-9A-Za-z]+]] = fir.convert [[Z2_ARR]] : (!fir.ref<!fir.array<2x!fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // AARCH64-DAG: [[Z2_VAL:%[0-9A-Za-z]+]] = fir.load [[Z2_ADDRC]] : !fir.ref<!fir.complex<4>>
  // AARCH64-DAG: [[Z1_ARR:%[0-9A-Za-z]+]] = fir.alloca !fir.array<2x!fir.real<4>>
  // AARCH64-DAG: fir.store [[Z1]] to [[Z1_ARR]] : !fir.ref<!fir.array<2x!fir.real<4>>>
  // AARCH64-DAG: [[Z1_ADDRC:%[0-9A-Za-z]+]] = fir.convert [[Z1_ARR]] : (!fir.ref<!fir.array<2x!fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // AARCH64-DAG: [[Z1_VAL:%[0-9A-Za-z]+]] = fir.load [[Z1_ADDRC]] : !fir.ref<!fir.complex<4>>

  // AARCH64-DAG: [[Z1_ARR2:%[0-9A-Za-z]+]] = fir.alloca !fir.array<2x!fir.real<4>>
  // AARCH64-DAG: [[Z1_ADDRC2:%[0-9A-Za-z]+]] = fir.convert [[Z1_ARR2]] : (!fir.ref<!fir.array<2x!fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // AARCH64-DAG: fir.store [[Z1_VAL]] to [[Z1_ADDRC2]] : !fir.ref<!fir.complex<4>>
  // AARCH64-DAG: [[Z1_RELOADED:%[0-9A-Za-z]+]] = fir.load [[Z1_ARR2]] : !fir.ref<!fir.array<2x!fir.real<4>>>
  // AARCH64-DAG: [[Z2_ARR2:%[0-9A-Za-z]+]] = fir.alloca !fir.array<2x!fir.real<4>>
  // AARCH64-DAG: [[Z2_ADDRC2:%[0-9A-Za-z]+]] = fir.convert [[Z2_ARR2]] : (!fir.ref<!fir.array<2x!fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // AARCH64-DAG: fir.store [[Z2_VAL]] to [[Z2_ADDRC2]] : !fir.ref<!fir.complex<4>>
  // AARCH64-DAG: [[Z2_RELOADED:%[0-9A-Za-z]+]] = fir.load [[Z2_ARR2]] : !fir.ref<!fir.array<2x!fir.real<4>>>
  // AARCH64-DAG: [[Z3_ARR2:%[0-9A-Za-z]+]] = fir.alloca !fir.array<2x!fir.real<4>>
  // AARCH64-DAG: [[Z3_ADDRC2:%[0-9A-Za-z]+]] = fir.convert [[Z3_ARR2]] : (!fir.ref<!fir.array<2x!fir.real<4>>>) -> !fir.ref<!fir.complex<4>>
  // AARCH64-DAG: fir.store [[Z3_VAL]] to [[Z3_ADDRC2]] : !fir.ref<!fir.complex<4>>
  // AARCH64-DAG: [[Z3_RELOADED:%[0-9A-Za-z]+]] = fir.load [[Z3_ARR2]] : !fir.ref<!fir.array<2x!fir.real<4>>>

  // AARCH64: fir.call @calleemultipleparamscomplex4([[Z1_RELOADED]], [[Z2_RELOADED]], [[Z3_RELOADED]]) : (!fir.array<2x!fir.real<4>>, !fir.array<2x!fir.real<4>>, !fir.array<2x!fir.real<4>>) -> ()

  // PPC-DAG: [[Z3_EMPTY:%[0-9A-Za-z]+]] = fir.undefined !fir.complex<4>
  // PPC-DAG: [[Z3_PARTIAL:%[0-9A-Za-z]+]] = fir.insert_value [[Z3_EMPTY]], [[A3]], [0 : i32] : (!fir.complex<4>, !fir.real<4>) -> !fir.complex<4>
  // PPC-DAG: [[Z3:%[0-9A-Za-z]+]] = fir.insert_value [[Z3_PARTIAL]], [[B3]], [1 : i32] : (!fir.complex<4>, !fir.real<4>) -> !fir.complex<4>
  // PPC-DAG: [[Z2_EMPTY:%[0-9A-Za-z]+]] = fir.undefined !fir.complex<4>
  // PPC-DAG: [[Z2_PARTIAL:%[0-9A-Za-z]+]] = fir.insert_value [[Z2_EMPTY]], [[A2]], [0 : i32] : (!fir.complex<4>, !fir.real<4>) -> !fir.complex<4>
  // PPC-DAG: [[Z2:%[0-9A-Za-z]+]] = fir.insert_value [[Z2_PARTIAL]], [[B2]], [1 : i32] : (!fir.complex<4>, !fir.real<4>) -> !fir.complex<4>
  // PPC-DAG: [[Z1_EMPTY:%[0-9A-Za-z]+]] = fir.undefined !fir.complex<4>
  // PPC-DAG: [[Z1_PARTIAL:%[0-9A-Za-z]+]] = fir.insert_value [[Z1_EMPTY]], [[A1]], [0 : i32] : (!fir.complex<4>, !fir.real<4>) -> !fir.complex<4>
  // PPC-DAG: [[Z1:%[0-9A-Za-z]+]] = fir.insert_value [[Z1_PARTIAL]], [[B1]], [1 : i32] : (!fir.complex<4>, !fir.real<4>) -> !fir.complex<4>

  // PPC-DAG: [[A1_EXTR:%[0-9A-Za-z]+]] = fir.extract_value [[Z1]], [0 : i32] : (!fir.complex<4>) -> !fir.real<4>
  // PPC-DAG: [[B1_EXTR:%[0-9A-Za-z]+]] = fir.extract_value [[Z1]], [1 : i32] : (!fir.complex<4>) -> !fir.real<4>
  // PPC-DAG: [[A2_EXTR:%[0-9A-Za-z]+]] = fir.extract_value [[Z2]], [0 : i32] : (!fir.complex<4>) -> !fir.real<4>
  // PPC-DAG: [[B2_EXTR:%[0-9A-Za-z]+]] = fir.extract_value [[Z2]], [1 : i32] : (!fir.complex<4>) -> !fir.real<4>
  // PPC-DAG: [[A3_EXTR:%[0-9A-Za-z]+]] = fir.extract_value [[Z3]], [0 : i32] : (!fir.complex<4>) -> !fir.real<4>
  // PPC-DAG: [[B3_EXTR:%[0-9A-Za-z]+]] = fir.extract_value [[Z3]], [1 : i32] : (!fir.complex<4>) -> !fir.real<4>

  // PPC: fir.call @calleemultipleparamscomplex4([[A1_EXTR]], [[B1_EXTR]], [[A2_EXTR]], [[B2_EXTR]], [[A3_EXTR]], [[B3_EXTR]]) : (!fir.real<4>, !fir.real<4>, !fir.real<4>, !fir.real<4>, !fir.real<4>, !fir.real<4>) -> ()

  fir.call @calleemultipleparamscomplex4(%z1, %z2, %z3) : (!fir.complex<4>, !fir.complex<4>, !fir.complex<4>) -> ()
  return
}

// Test that we rewrite the signature of and calls to a function that accepts
// and returns MLIR complex<f32>.

// I32-LABEL: func private @mlircomplexf32
// I32-SAME: ([[Z1:%[0-9A-Za-z]+]]: !fir.ref<tuple<f32, f32>> {llvm.align = 4 : i32, llvm.byval = tuple<f32, f32>}, [[Z2:%[0-9A-Za-z]+]]: !fir.ref<tuple<f32, f32>> {llvm.align = 4 : i32, llvm.byval = tuple<f32, f32>})
// I32-SAME: -> i64
// X64-LABEL: func private @mlircomplexf32
// X64-SAME: ([[Z1:%[0-9A-Za-z]+]]: !fir.vector<2:f32>, [[Z2:%[0-9A-Za-z]+]]: !fir.vector<2:f32>)
// X64-SAME: -> !fir.vector<2:f32>
// AARCH64-LABEL: func private @mlircomplexf32
// AARCH64-SAME: ([[Z1:%[0-9A-Za-z]+]]: !fir.array<2xf32>, [[Z2:%[0-9A-Za-z]+]]: !fir.array<2xf32>)
// AARCH64-SAME: -> tuple<f32, f32>
// PPC-LABEL: func private @mlircomplexf32
// PPC-SAME: ([[A1:%[0-9A-Za-z]+]]: f32, [[B1:%[0-9A-Za-z]+]]: f32, [[A2:%[0-9A-Za-z]+]]: f32, [[B2:%[0-9A-Za-z]+]]: f32)
// PPC-SAME: -> tuple<f32, f32>
func.func private @mlircomplexf32(%z1: complex<f32>, %z2: complex<f32>) -> complex<f32> {

  // I32-DAG: [[Z1_ADDR:%[0-9A-Za-z]+]] = fir.convert [[Z1]] : (!fir.ref<tuple<f32, f32>>) -> !fir.ref<complex<f32>>
  // I32-DAG: [[Z1_VAL:%[0-9A-Za-z]+]] = fir.load [[Z1_ADDR]] : !fir.ref<complex<f32>>
  // I32-DAG: [[Z2_ADDR:%[0-9A-Za-z]+]] = fir.convert [[Z2]] : (!fir.ref<tuple<f32, f32>>) -> !fir.ref<complex<f32>>
  // I32-DAG: [[Z2_VAL:%[0-9A-Za-z]+]] = fir.load [[Z2_ADDR]] : !fir.ref<complex<f32>>

  // I32-DAG: [[Z1_ADDRC:%[0-9A-Za-z]+]] = fir.alloca complex<f32>
  // I32-DAG: fir.store [[Z1_VAL]] to [[Z1_ADDRC]] : !fir.ref<complex<f32>>
  // I32-DAG: [[Z1_ADDRT:%[0-9A-Za-z]+]] = fir.convert [[Z1_ADDRC]] : (!fir.ref<complex<f32>>) -> !fir.ref<tuple<f32, f32>>
  // I32-DAG: [[Z2_ADDRC:%[0-9A-Za-z]+]] = fir.alloca complex<f32>
  // I32-DAG: fir.store [[Z2_VAL]] to [[Z2_ADDRC]] : !fir.ref<complex<f32>>
  // I32-DAG: [[Z2_ADDRT:%[0-9A-Za-z]+]] = fir.convert [[Z2_ADDRC]] : (!fir.ref<complex<f32>>) -> !fir.ref<tuple<f32, f32>>

  // I32: [[VAL:%[0-9A-Za-z]+]] = fir.call @mlircomplexf32([[Z1_ADDRT]], [[Z2_ADDRT]]) : (!fir.ref<tuple<f32, f32>>, !fir.ref<tuple<f32, f32>>) -> i64

  // X64-DAG: [[Z2_ADDR:%[0-9A-Za-z]+]] = fir.alloca !fir.vector<2:f32>
  // X64-DAG: fir.store [[Z2]] to [[Z2_ADDR]] : !fir.ref<!fir.vector<2:f32>>
  // X64-DAG: [[Z2_ADDRC:%[0-9A-Za-z]+]] = fir.convert [[Z2_ADDR]] : (!fir.ref<!fir.vector<2:f32>>) -> !fir.ref<complex<f32>>
  // X64-DAG: [[Z2_VAL:%[0-9A-Za-z]+]] = fir.load [[Z2_ADDRC]] : !fir.ref<complex<f32>>
  // X64-DAG: [[Z1_ADDR:%[0-9A-Za-z]+]] = fir.alloca !fir.vector<2:f32>
  // X64-DAG: fir.store [[Z1]] to [[Z1_ADDR]] : !fir.ref<!fir.vector<2:f32>>
  // X64-DAG: [[Z1_ADDRC:%[0-9A-Za-z]+]] = fir.convert [[Z1_ADDR]] : (!fir.ref<!fir.vector<2:f32>>) -> !fir.ref<complex<f32>>
  // X64-DAG: [[Z1_VAL:%[0-9A-Za-z]+]] = fir.load [[Z1_ADDRC]] : !fir.ref<complex<f32>>

  // X64-DAG: [[Z1_ADDRV:%[0-9A-Za-z]+]] = fir.alloca !fir.vector<2:f32>
  // X64-DAG: [[Z1_ADDRC2:%[0-9A-Za-z]+]] = fir.convert [[Z1_ADDRV]] : (!fir.ref<!fir.vector<2:f32>>) -> !fir.ref<complex<f32>>
  // X64-DAG: fir.store [[Z1_VAL]] to [[Z1_ADDRC2]] : !fir.ref<complex<f32>>
  // X64-DAG: [[Z1_RELOADED:%[0-9A-Za-z]+]] = fir.load [[Z1_ADDRV]] : !fir.ref<!fir.vector<2:f32>>
  // X64-DAG: [[Z2_ADDRV:%[0-9A-Za-z]+]] = fir.alloca !fir.vector<2:f32>
  // X64-DAG: [[Z2_ADDRC2:%[0-9A-Za-z]+]] = fir.convert [[Z2_ADDRV]] : (!fir.ref<!fir.vector<2:f32>>) -> !fir.ref<complex<f32>>
  // X64-DAG: fir.store [[Z2_VAL]] to [[Z2_ADDRC2]] : !fir.ref<complex<f32>>
  // X64-DAG: [[Z2_RELOADED:%[0-9A-Za-z]+]] = fir.load [[Z2_ADDRV]] : !fir.ref<!fir.vector<2:f32>>

  // X64: [[VAL:%[0-9A-Za-z]+]] = fir.call @mlircomplexf32([[Z1_RELOADED]], [[Z2_RELOADED]]) : (!fir.vector<2:f32>, !fir.vector<2:f32>) -> !fir.vector<2:f32>

  // AARCH64-DAG: [[Z2_ARR:%[0-9A-Za-z]+]] = fir.alloca !fir.array<2xf32>
  // AARCH64-DAG: fir.store [[Z2]] to [[Z2_ARR]] : !fir.ref<!fir.array<2xf32>>
  // AARCH64-DAG: [[Z2_ADDRC:%[0-9A-Za-z]+]] = fir.convert [[Z2_ARR]] : (!fir.ref<!fir.array<2xf32>>) -> !fir.ref<complex<f32>>
  // AARCH64-DAG: [[Z2_VAL:%[0-9A-Za-z]+]] = fir.load [[Z2_ADDRC]] : !fir.ref<complex<f32>>
  // AARCH64-DAG: [[Z1_ARR:%[0-9A-Za-z]+]] = fir.alloca !fir.array<2xf32>
  // AARCH64-DAG: fir.store [[Z1]] to [[Z1_ARR]] : !fir.ref<!fir.array<2xf32>>
  // AARCH64-DAG: [[Z1_ADDRC:%[0-9A-Za-z]+]] = fir.convert [[Z1_ARR]] : (!fir.ref<!fir.array<2xf32>>) -> !fir.ref<complex<f32>>
  // AARCH64-DAG: [[Z1_VAL:%[0-9A-Za-z]+]] = fir.load [[Z1_ADDRC]] : !fir.ref<complex<f32>>

  // AARCH64-DAG: [[Z1_ARR2:%[0-9A-Za-z]+]] = fir.alloca !fir.array<2xf32>
  // AARCH64-DAG: [[Z1_ADDRC2:%[0-9A-Za-z]+]] = fir.convert [[Z1_ARR2]] : (!fir.ref<!fir.array<2xf32>>) -> !fir.ref<complex<f32>>
  // AARCH64-DAG: fir.store [[Z1_VAL]] to [[Z1_ADDRC2]] : !fir.ref<complex<f32>>
  // AARCH64-DAG: [[Z1_RELOADED:%[0-9A-Za-z]+]] = fir.load [[Z1_ARR2]] : !fir.ref<!fir.array<2xf32>>
  // AARCH64-DAG: [[Z2_ARR2:%[0-9A-Za-z]+]] = fir.alloca !fir.array<2xf32>
  // AARCH64-DAG: [[Z2_ADDRC2:%[0-9A-Za-z]+]] = fir.convert [[Z2_ARR2]] : (!fir.ref<!fir.array<2xf32>>) -> !fir.ref<complex<f32>>
  // AARCH64-DAG: fir.store [[Z2_VAL]] to [[Z2_ADDRC2]] : !fir.ref<complex<f32>>
  // AARCH64-DAG: [[Z2_RELOADED:%[0-9A-Za-z]+]] = fir.load [[Z2_ARR2]] : !fir.ref<!fir.array<2xf32>>

  // AARCH64: [[VAL:%[0-9A-Za-z]+]] = fir.call @mlircomplexf32([[Z1_RELOADED]], [[Z2_RELOADED]]) : (!fir.array<2xf32>, !fir.array<2xf32>) -> tuple<f32, f32>

  // PPC-DAG: [[Z2_EMPTY:%[0-9A-Za-z]+]] = fir.undefined complex<f32>
  // PPC-DAG: [[Z2_PARTIAL:%[0-9A-Za-z]+]] = fir.insert_value [[Z2_EMPTY]], [[A2]], [0 : i32] : (complex<f32>, f32) -> complex<f32>
  // PPC-DAG: [[Z2:%[0-9A-Za-z]+]] = fir.insert_value [[Z2_PARTIAL]], [[B2]], [1 : i32] : (complex<f32>, f32) -> complex<f32>
  // PPC-DAG: [[Z1_EMPTY:%[0-9A-Za-z]+]] = fir.undefined complex<f32>
  // PPC-DAG: [[Z1_PARTIAL:%[0-9A-Za-z]+]] = fir.insert_value [[Z1_EMPTY]], [[A1]], [0 : i32] : (complex<f32>, f32) -> complex<f32>
  // PPC-DAG: [[Z1:%[0-9A-Za-z]+]] = fir.insert_value [[Z1_PARTIAL]], [[B1]], [1 : i32] : (complex<f32>, f32) -> complex<f32>

  // PPC-DAG: [[A1_EXTR:%[0-9A-Za-z]+]] = fir.extract_value [[Z1]], [0 : i32] : (complex<f32>) -> f32
  // PPC-DAG: [[B1_EXTR:%[0-9A-Za-z]+]] = fir.extract_value [[Z1]], [1 : i32] : (complex<f32>) -> f32
  // PPC-DAG: [[A2_EXTR:%[0-9A-Za-z]+]] = fir.extract_value [[Z2]], [0 : i32] : (complex<f32>) -> f32
  // PPC-DAG: [[B2_EXTR:%[0-9A-Za-z]+]] = fir.extract_value [[Z2]], [1 : i32] : (complex<f32>) -> f32

  // PPC: [[VAL:%[0-9A-Za-z]+]] = fir.call @mlircomplexf32([[A1_EXTR]], [[B1_EXTR]], [[A2_EXTR]], [[B2_EXTR]]) : (f32, f32, f32, f32) -> tuple<f32, f32>
  %0 = fir.call @mlircomplexf32(%z1, %z2) : (complex<f32>, complex<f32>) -> complex<f32>


  // I32: [[ADDRI64:%[0-9A-Za-z]+]] = fir.alloca i64
  // I32: fir.store [[VAL]] to [[ADDRI64]] : !fir.ref<i64>
  // I32: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRI64]] : (!fir.ref<i64>) -> !fir.ref<complex<f32>>
  // I32: [[VAL_2:%[0-9A-Za-z]+]] = fir.load [[ADDRC]] : !fir.ref<complex<f32>>
  // I32: [[ADDRI64_2:%[0-9A-Za-z]+]] = fir.alloca i64
  // I32: [[ADDRC_2:%[0-9A-Za-z]+]] = fir.convert [[ADDRI64_2]] : (!fir.ref<i64>) -> !fir.ref<complex<f32>>
  // I32: fir.store [[VAL_2]] to [[ADDRC_2]] : !fir.ref<complex<f32>>
  // I32: [[RES:%[0-9A-Za-z]+]] = fir.load [[ADDRI64_2]] : !fir.ref<i64>
  // I32: return [[RES]] : i64

  // X64: [[ADDRV:%[0-9A-Za-z]+]] = fir.alloca !fir.vector<2:f32>
  // X64: fir.store [[VAL]] to [[ADDRV]] : !fir.ref<!fir.vector<2:f32>>
  // X64: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRV]] : (!fir.ref<!fir.vector<2:f32>>) -> !fir.ref<complex<f32>>
  // X64: [[V:%[0-9A-Za-z]+]] = fir.load [[ADDRC]] : !fir.ref<complex<f32>>
  // X64: [[ADDRV_2:%[0-9A-Za-z]+]] = fir.alloca !fir.vector<2:f32>
  // X64: [[ADDRC_2:%[0-9A-Za-z]+]] = fir.convert [[ADDRV_2]] : (!fir.ref<!fir.vector<2:f32>>) -> !fir.ref<complex<f32>>
  // X64: fir.store [[V]] to [[ADDRC_2]] : !fir.ref<complex<f32>>
  // X64: [[RES:%[0-9A-Za-z]+]] = fir.load [[ADDRV_2]] : !fir.ref<!fir.vector<2:f32>>
  // X64: return [[RES]] : !fir.vector<2:f32>

  // AARCH64: [[ADDRT:%[0-9A-Za-z]+]] = fir.alloca tuple<f32, f32>
  // AARCH64: fir.store [[VAL]] to [[ADDRT]] : !fir.ref<tuple<f32, f32>>
  // AARCH64: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRT]] : (!fir.ref<tuple<f32, f32>>) -> !fir.ref<complex<f32>>
  // AARCH64: [[V:%[0-9A-Za-z]+]] = fir.load [[ADDRC]] : !fir.ref<complex<f32>>
  // AARCH64: [[ADDRT_2:%[0-9A-Za-z]+]] = fir.alloca tuple<f32, f32>
  // AARCH64: [[ADDRC_2:%[0-9A-Za-z]+]] = fir.convert [[ADDRT_2]] : (!fir.ref<tuple<f32, f32>>) -> !fir.ref<complex<f32>>
  // AARCH64: fir.store [[V]] to [[ADDRC_2]] : !fir.ref<complex<f32>>
  // AARCH64: [[RES:%[0-9A-Za-z]+]] = fir.load [[ADDRT_2]] : !fir.ref<tuple<f32, f32>>
  // AARCH64: return [[RES]] : tuple<f32, f32>

  // PPC: [[ADDRT:%[0-9A-Za-z]+]] = fir.alloca tuple<f32, f32>
  // PPC: fir.store [[VAL]] to [[ADDRT]] : !fir.ref<tuple<f32, f32>>
  // PPC: [[ADDRC:%[0-9A-Za-z]+]] = fir.convert [[ADDRT]] : (!fir.ref<tuple<f32, f32>>) -> !fir.ref<complex<f32>>
  // PPC: [[V:%[0-9A-Za-z]+]] = fir.load [[ADDRC]] : !fir.ref<complex<f32>>
  // PPC: [[ADDRT_2:%[0-9A-Za-z]+]] = fir.alloca tuple<f32, f32>
  // PPC: [[ADDRC_2:%[0-9A-Za-z]+]] = fir.convert [[ADDRT_2]] : (!fir.ref<tuple<f32, f32>>) -> !fir.ref<complex<f32>>
  // PPC: fir.store [[V]] to [[ADDRC_2]] : !fir.ref<complex<f32>>
  // PPC: [[RES:%[0-9A-Za-z]+]] = fir.load [[ADDRT_2]] : !fir.ref<tuple<f32, f32>>
  // PPC: return [[RES]] : tuple<f32, f32>
  return %0 : complex<f32>
}

// Test that we rewrite the fir.address_of operator.
// I32-LABEL: func @addrof()
// X64-LABEL: func @addrof()
// AARCH64-LABEL: func @addrof()
// PPC-LABEL: func @addrof()
func.func @addrof() {
  // I32: {{%.*}} = fir.address_of(@returncomplex4) : () -> i64
  // X64: {{%.*}} = fir.address_of(@returncomplex4) : () -> !fir.vector<2:!fir.real<4>>
  // AARCH64: {{%.*}} = fir.address_of(@returncomplex4) : () -> tuple<!fir.real<4>, !fir.real<4>>
  // PPC: {{%.*}} = fir.address_of(@returncomplex4) : () -> tuple<!fir.real<4>, !fir.real<4>>
  %r = fir.address_of(@returncomplex4) : () -> !fir.complex<4>

  // I32: {{%.*}} = fir.address_of(@paramcomplex4) : (!fir.ref<tuple<!fir.real<4>, !fir.real<4>>>) -> ()
  // X64: {{%.*}} = fir.address_of(@paramcomplex4) : (!fir.vector<2:!fir.real<4>>) -> ()
  // AARCH64: {{%.*}} = fir.address_of(@paramcomplex4) : (!fir.array<2x!fir.real<4>>) -> ()
  // PPC: {{%.*}} = fir.address_of(@paramcomplex4) : (!fir.real<4>, !fir.real<4>) -> ()
  %p = fir.address_of(@paramcomplex4) : (!fir.complex<4>) -> ()
  return
}
