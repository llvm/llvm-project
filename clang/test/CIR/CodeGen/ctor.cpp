// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Struk {
  int a;
  Struk();
};

void baz() {
  Struk s;
}

// CHECK: !rec_Struk = !cir.record<struct "Struk" {!s32i}>

// CHECK:   cir.func @_ZN5StrukC1Ev(!cir.ptr<!rec_Struk>)
// CHECK:   cir.func @_Z3bazv()
// CHECK-NEXT:     %[[S_ADDR:.*]] = cir.alloca !rec_Struk, !cir.ptr<!rec_Struk>, ["s", init] {alignment = 4 : i64}
// CHECK-NEXT:     cir.call @_ZN5StrukC1Ev(%[[S_ADDR]]) : (!cir.ptr<!rec_Struk>) -> ()
// CHECK-NEXT:     cir.return
