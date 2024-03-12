// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck -check-prefix=BEFORE %s

void foo() noexcept;

class xpto {
public:
  xpto() {
    foo();
  }
  int i;
  float f;
  ~xpto() {
    foo();
  }
};

void x() {
  xpto array[2];
}

// BEFORE: cir.func @_Z1xv()
// BEFORE:   %[[ArrayAddr:.*]] = cir.alloca !cir.array<!ty_22xpto22 x 2>

// BEFORE:   cir.array.ctor(%[[ArrayAddr]] : !cir.ptr<!cir.array<!ty_22xpto22 x 2>>) {
// BEFORE:   ^bb0(%arg0: !cir.ptr<!ty_22xpto22>
// BEFORE:     cir.call @_ZN4xptoC1Ev(%arg0) : (!cir.ptr<!ty_22xpto22>) -> ()
// BEFORE:     cir.yield
// BEFORE:   }

// BEFORE:   cir.array.dtor(%[[ArrayAddr]] : !cir.ptr<!cir.array<!ty_22xpto22 x 2>>) {
// BEFORE:   ^bb0(%arg0: !cir.ptr<!ty_22xpto22>
// BEFORE:     cir.call @_ZN4xptoD1Ev(%arg0) : (!cir.ptr<!ty_22xpto22>) -> ()
// BEFORE:     cir.yield
// BEFORE:   }
