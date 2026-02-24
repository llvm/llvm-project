// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir -fcxx-exceptions -fexceptions
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR

struct StructWithDestructor {
  ~StructWithDestructor();
  void procedure();
};

void cleanup_scope_with_without_body() { StructWithDestructor a; }

// CIR: %[[A_ADDR:.*]] = cir.alloca !rec_StructWithDestructor, !cir.ptr<!rec_StructWithDestructor>, ["a"]
// CIR: cir.cleanup.scope {
// CIR:   cir.yield
// CIR: } cleanup all {
// CIR:   cir.call @_ZN20StructWithDestructorD1Ev(%[[A_ADDR]]) nothrow : (!cir.ptr<!rec_StructWithDestructor>) -> ()
// CIR:   cir.yield
// CIR: }

void cleanup_scope_with_body_and_cleanup() {
  StructWithDestructor a;
  a.procedure();
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !rec_StructWithDestructor, !cir.ptr<!rec_StructWithDestructor>, ["a"]
// CIR: cir.cleanup.scope {
// CIR:   cir.call @_ZN20StructWithDestructor9procedureEv(%[[A_ADDR]]) : (!cir.ptr<!rec_StructWithDestructor>) -> ()
// CIR:   cir.yield
// CIR: } cleanup all {
// CIR:   cir.call @_ZN20StructWithDestructorD1Ev(%0) nothrow : (!cir.ptr<!rec_StructWithDestructor>) -> ()
// CIR:   cir.yield
// CIR: }
