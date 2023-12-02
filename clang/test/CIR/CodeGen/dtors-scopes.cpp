// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -clangir-disable-emit-cxx-default -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -std=c++20 -fclangir -emit-cir %s -o %t2.cir
// RUN: FileCheck --input-file=%t2.cir %s --check-prefix=DTOR_BODY

extern "C" int printf(char const*, ...);
struct C {
  C()  { printf("++A\n"); }
  ~C()  { printf("--A\n"); }
};
void dtor1() {
  {
    C c;
  }
  printf("Done\n");
}

// CHECK: cir.func @_Z5dtor1v()
// CHECK:   cir.scope {
// CHECK:     %4 = cir.alloca !ty_22C22, cir.ptr <!ty_22C22>, ["c", init] {alignment = 1 : i64}
// CHECK:     cir.call @_ZN1CC2Ev(%4) : (!cir.ptr<!ty_22C22>) -> ()
// CHECK:     cir.call @_ZN1CD2Ev(%4) : (!cir.ptr<!ty_22C22>) -> ()
// CHECK:   }

// DTOR_BODY: cir.func linkonce_odr @_ZN1CD2Ev{{.*}}{
// DTOR_BODY:   %2 = cir.get_global @printf
// DTOR_BODY:   %3 = cir.get_global @".str2"
// DTOR_BODY:   %4 = cir.cast(array_to_ptrdecay, %3
// DTOR_BODY:   %5 = cir.call @printf(%4)
// DTOR_BODY:   cir.return

// DTOR_BODY: cir.func linkonce_odr @_ZN1CD1Ev(%arg0: !cir.ptr<!ty_22C22>

// DTOR_BODY:   cir.call @_ZN1CD2Ev
// DTOR_BODY:   cir.return
// DTOR_BODY: }