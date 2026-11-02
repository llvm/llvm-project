// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefixes=CIR,ASM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -fcxx-exceptions -fexceptions -emit-cir %s -o %t.eh.cir
// RUN: FileCheck --input-file=%t.eh.cir %s --check-prefix=ASM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=ASM-IR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=ASM-IR

struct StructWithDestructor {
  ~StructWithDestructor();
  void procedure();
};

void cleanup_scope_with_without_body() { StructWithDestructor a; }

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!rec_StructWithDestructor>
// CIR: cir.cleanup.scope {
// CIR:   cir.yield
// CIR: } cleanup normal {
// CIR:   cir.call @_ZN20StructWithDestructorD1Ev(%[[A_ADDR]]) nothrow : (!cir.ptr<!rec_StructWithDestructor> {{.*}}) -> ()
// CIR:   cir.yield
// CIR: }

void cleanup_scope_with_body_and_cleanup() {
  StructWithDestructor a;
  a.procedure();
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!rec_StructWithDestructor>
// CIR: cir.cleanup.scope {
// CIR:   cir.call @_ZN20StructWithDestructor9procedureEv(%[[A_ADDR]]) : (!cir.ptr<!rec_StructWithDestructor> {{.*}}) -> ()
// CIR:   cir.yield
// CIR: } cleanup normal {
// CIR:   cir.call @_ZN20StructWithDestructorD1Ev(%0) nothrow : (!cir.ptr<!rec_StructWithDestructor> {{.*}}) -> ()
// CIR:   cir.yield
// CIR: }

// The operands of an asm statement are not wrapped in an ExprWithCleanups.
// Their temporaries are destroyed at the end of the asm statement.

struct S {
  S(int);
  ~S();
};

extern "C" {

int g(const S &);
void use(int);

void asm_plain() {
  asm volatile("" : : "r"(g(S(1))));
  use(0);
}

// ASM-LABEL: cir.func {{.*}} @asm_plain(
// ASM:         %[[TMP:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_S>
// ASM:         cir.call @_ZN1SC1Ei(%[[TMP]],
// ASM-NEXT:    cir.cleanup.scope {
// ASM-NEXT:      %[[G:.*]] = cir.call @g(%[[TMP]])
// ASM-NEXT:      cir.asm(
// ASM:             in = [%[[G]] : !s32i],
// ASM:           cir.yield
// ASM-NEXT:    } cleanup {{(normal|all)}} {
// ASM-NEXT:      cir.call @_ZN1SD1Ev(%[[TMP]])
// ASM-NEXT:      cir.yield
// ASM-NEXT:    }
// ASM-NEXT:    %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// ASM-NEXT:    cir.call @use(%[[ZERO]])

// ASM-IR-LABEL: define {{.*}} void @asm_plain(
// ASM-IR:         call void @_ZN1SC1Ei(ptr {{.*}} %[[TMP:.*]], i32 {{.*}} 1)
// ASM-IR:         call void asm sideeffect
// ASM-IR:         call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// ASM-IR:         call void @use(i32 {{.*}} 0)

// The cleanup scope of the enclosing full-expression is open, but the asm
// statement in the statement expression needs its own.
void asm_in_statement_expression(bool b) {
  use(g(S(3)) + (b ? 1 : 0) + ({
        int y = 0;
        asm volatile("" : "+r"(y) : "r"(b ? g(S(4)) : 0));
        y;
      }));
}

// ASM-LABEL: cir.func {{.*}} @asm_in_statement_expression(
// ASM:         %[[TMP3:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_S>
// ASM:         %[[TMP4:.*]] = cir.alloca "ref.tmp1" {{.*}} : !cir.ptr<!rec_S>
// ASM:         %[[ACTIVE:.*]] = cir.alloca "cleanup.cond" {{.*}} : !cir.ptr<!cir.bool>
// ASM:         cir.scope {
// ASM:           cir.cleanup.scope {
// ASM-NEXT:        %[[FALSE:.*]] = cir.const #false
// ASM-NEXT:        cir.store %[[FALSE]], %[[ACTIVE]]
// ASM-NEXT:        %[[ARG:.*]] = cir.ternary(%{{.*}}, true {
// ASM:               cir.call @_ZN1SC1Ei(%[[TMP4]],
// ASM-NEXT:          %[[TRUE:.*]] = cir.const #true
// ASM-NEXT:          cir.store %[[TRUE]], %[[ACTIVE]]
// ASM:               cir.call @g(%[[TMP4]])
// ASM:             }) : (!cir.bool) -> !s32i
// ASM-NEXT:        cir.asm(
// ASM:               in = [%[[ARG]] : !s32i],
// ASM:             cir.yield
// ASM-NEXT:      } cleanup {{(normal|all)}} {
// ASM-NEXT:        %[[IS_ACTIVE:.*]] = cir.load {{.*}} %[[ACTIVE]]
// ASM-NEXT:        cir.if %[[IS_ACTIVE]] {
// ASM-NEXT:          cir.call @_ZN1SD1Ev(%[[TMP4]]){{.*}}
// ASM-NEXT:        }
// ASM-NEXT:        cir.yield
// ASM-NEXT:      }
// ASM:         }
// ASM:         cir.call @use(
// ASM:       } cleanup {{(normal|all)}} {
// ASM-NEXT:    cir.call @_ZN1SD1Ev(%[[TMP3]])

// ASM-IR-LABEL: define {{.*}} void @asm_in_statement_expression(
// ASM-IR:         call void @_ZN1SC1Ei(ptr {{.*}} %[[TMP3:.*]], i32 {{.*}} 3)
// ASM-IR:         call void @_ZN1SC1Ei(ptr {{.*}} %[[TMP4:.*]], i32 {{.*}} 4)
// ASM-IR:         call i32 asm sideeffect
// ASM-IR:         call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP4]])
// ASM-IR:         call void @use(
// ASM-IR:         call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP3]])

} // extern "C"
