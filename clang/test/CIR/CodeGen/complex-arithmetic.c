// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -complex-range=basic -fclangir -clangir-disable-passes -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir --check-prefixes=CLANG,CIRGEN,CIRGEN-BASIC,CHECK %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -complex-range=basic -fclangir -clangir-disable-passes -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir --check-prefixes=CPPLANG,CIRGEN,CIRGEN-BASIC,CHECK %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -complex-range=improved -fclangir -clangir-disable-passes -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir --check-prefixes=CLANG,CIRGEN,CIRGEN-IMPROVED,CHECK %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -complex-range=improved -fclangir -clangir-disable-passes -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir --check-prefixes=CPPLANG,CIRGEN,CIRGEN-IMPROVED,CHECK %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -complex-range=full -fclangir -clangir-disable-passes -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir --check-prefixes=CLANG,CIRGEN,CIRGEN-FULL,CHECK %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -complex-range=full -fclangir -clangir-disable-passes -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir --check-prefixes=CPPLANG,CIRGEN,CIRGEN-FULL,CHECK %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -complex-range=basic -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir --check-prefixes=CLANG,CIR,CIR-BASIC,CHECK %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -complex-range=basic -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir --check-prefixes=CPPLANG,CIR,CIR-BASIC,CHECK %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -complex-range=improved -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir --check-prefixes=CLANG,CIR,CIR-IMPROVED,CHECK %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -complex-range=improved -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir --check-prefixes=CPPLANG,CIR,CIR-IMPROVED,CHECK %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -complex-range=full -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir --check-prefixes=CLANG,CIR,CIR-FULL,CHECK %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -complex-range=full -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir --check-prefixes=CPPLANG,CIR,CIR-FULL,CHECK %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -complex-range=basic -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll --check-prefixes=CLANG,LLVM,LLVM-BASIC,CHECK %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -complex-range=basic -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll --check-prefixes=CPPLANG,LLVM,LLVM-BASIC,CHECK %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -complex-range=improved -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll --check-prefixes=CLANG,LLVM,LLVM-IMPROVED,CHECK %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -complex-range=improved -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll --check-prefixes=CPPLANG,LLVM,LLVM-IMPROVED,CHECK %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -complex-range=full -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll --check-prefixes=CLANG,LLVM,LLVM-FULL,CHECK %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -complex-range=full -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll --check-prefixes=CPPLANG,LLVM,LLVM-FULL,CHECK %s

double _Complex cd1, cd2;
int _Complex ci1, ci2;

void add() {
  cd1 = cd1 + cd2;
  ci1 = ci1 + ci2;
}

// CLANG:   @add
// CPPLANG: @_Z3addv

// CIRGEN: %{{.+}} = cir.binop(add, %{{.+}}, %{{.+}}) : !cir.complex<!cir.double>
// CIRGEN: %{{.+}} = cir.binop(add, %{{.+}}, %{{.+}}) : !cir.complex<!s32i>

//      CIR: %[[#LHS_REAL:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#LHS_IMAG:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#RHS_REAL:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#RHS_IMAG:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#RES_REAL:]] = cir.binop(add, %[[#LHS_REAL]], %[[#RHS_REAL]]) : !cir.double
// CIR-NEXT: %[[#RES_IMAG:]] = cir.binop(add, %[[#LHS_IMAG]], %[[#RHS_IMAG]]) : !cir.double
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#RES_REAL]], %[[#RES_IMAG]] : !cir.double -> !cir.complex<!cir.double>

//      CIR: %[[#LHS_REAL:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#LHS_IMAG:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#RHS_REAL:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#RHS_IMAG:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#RES_REAL:]] = cir.binop(add, %[[#LHS_REAL]], %[[#RHS_REAL]]) : !s32i
// CIR-NEXT: %[[#RES_IMAG:]] = cir.binop(add, %[[#LHS_IMAG]], %[[#RHS_IMAG]]) : !s32i
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#RES_REAL]], %[[#RES_IMAG]] : !s32i -> !cir.complex<!s32i>

//      LLVM: %[[#LHS_REAL:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-NEXT: %[[#LHS_IMAG:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-NEXT: %[[#RHS_REAL:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-NEXT: %[[#RHS_IMAG:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-NEXT: %[[#RES_REAL:]] = fadd double %[[#LHS_REAL]], %[[#RHS_REAL]]
// LLVM-NEXT: %[[#RES_IMAG:]] = fadd double %[[#LHS_IMAG]], %[[#RHS_IMAG]]
// LLVM-NEXT: %[[#A:]] = insertvalue { double, double } undef, double %[[#RES_REAL]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { double, double } %[[#A]], double %[[#RES_IMAG]], 1

//      LLVM: %[[#LHS_REAL:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-NEXT: %[[#LHS_IMAG:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-NEXT: %[[#RHS_REAL:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-NEXT: %[[#RHS_IMAG:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-NEXT: %[[#RES_REAL:]] = add i32 %[[#LHS_REAL]], %[[#RHS_REAL]]
// LLVM-NEXT: %[[#RES_IMAG:]] = add i32 %[[#LHS_IMAG]], %[[#RHS_IMAG]]
// LLVM-NEXT: %[[#A:]] = insertvalue { i32, i32 } undef, i32 %[[#RES_REAL]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#A]], i32 %[[#RES_IMAG]], 1

// CHECK: }

void sub() {
  cd1 = cd1 - cd2;
  ci1 = ci1 - ci2;
}

// CLANG:   @sub
// CPPLANG: @_Z3subv

// CIRGEN: %{{.+}} = cir.binop(sub, %{{.+}}, %{{.+}}) : !cir.complex<!cir.double>
// CIRGEN: %{{.+}} = cir.binop(sub, %{{.+}}, %{{.+}}) : !cir.complex<!s32i>

//      CIR: %[[#LHS_REAL:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#LHS_IMAG:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#RHS_REAL:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#RHS_IMAG:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#RES_REAL:]] = cir.binop(sub, %[[#LHS_REAL]], %[[#RHS_REAL]]) : !cir.double
// CIR-NEXT: %[[#RES_IMAG:]] = cir.binop(sub, %[[#LHS_IMAG]], %[[#RHS_IMAG]]) : !cir.double
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#RES_REAL]], %[[#RES_IMAG]] : !cir.double -> !cir.complex<!cir.double>

//      CIR: %[[#LHS_REAL:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#LHS_IMAG:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#RHS_REAL:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#RHS_IMAG:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#RES_REAL:]] = cir.binop(sub, %[[#LHS_REAL]], %[[#RHS_REAL]]) : !s32i
// CIR-NEXT: %[[#RES_IMAG:]] = cir.binop(sub, %[[#LHS_IMAG]], %[[#RHS_IMAG]]) : !s32i
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#RES_REAL]], %[[#RES_IMAG]] : !s32i -> !cir.complex<!s32i>

//      LLVM: %[[#LHS_REAL:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-NEXT: %[[#LHS_IMAG:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-NEXT: %[[#RHS_REAL:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-NEXT: %[[#RHS_IMAG:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-NEXT: %[[#RES_REAL:]] = fsub double %[[#LHS_REAL]], %[[#RHS_REAL]]
// LLVM-NEXT: %[[#RES_IMAG:]] = fsub double %[[#LHS_IMAG]], %[[#RHS_IMAG]]
// LLVM-NEXT: %[[#A:]] = insertvalue { double, double } undef, double %[[#RES_REAL]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { double, double } %[[#A]], double %[[#RES_IMAG]], 1

//      LLVM: %[[#LHS_REAL:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-NEXT: %[[#LHS_IMAG:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-NEXT: %[[#RHS_REAL:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-NEXT: %[[#RHS_IMAG:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-NEXT: %[[#RES_REAL:]] = sub i32 %[[#LHS_REAL]], %[[#RHS_REAL]]
// LLVM-NEXT: %[[#RES_IMAG:]] = sub i32 %[[#LHS_IMAG]], %[[#RHS_IMAG]]
// LLVM-NEXT: %[[#A:]] = insertvalue { i32, i32 } undef, i32 %[[#RES_REAL]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#A]], i32 %[[#RES_IMAG]], 1

// CHECK: }

void mul() {
  cd1 = cd1 * cd2;
  ci1 = ci1 * ci2;
}

// CLANG:   @mul
// CPPLANG: @_Z3mulv

// CIRGEN-BASIC: %{{.+}} = cir.complex.binop mul %{{.+}}, %{{.+}} range(basic) : !cir.complex<!cir.double>
// CIRGEN-BASIC: %{{.+}} = cir.complex.binop mul %{{.+}}, %{{.+}} range(basic) : !cir.complex<!s32i>

//      CIR-BASIC: %[[#LHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-BASIC-NEXT: %[[#LHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-BASIC-NEXT: %[[#RHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-BASIC-NEXT: %[[#RHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-BASIC-NEXT: %[[#A:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSR]]) : !cir.double
// CIR-BASIC-NEXT: %[[#B:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSI]]) : !cir.double
// CIR-BASIC-NEXT: %[[#C:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSI]]) : !cir.double
// CIR-BASIC-NEXT: %[[#D:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSR]]) : !cir.double
// CIR-BASIC-NEXT: %[[#E:]] = cir.binop(sub, %[[#A]], %[[#B]]) : !cir.double
// CIR-BASIC-NEXT: %[[#F:]] = cir.binop(add, %[[#C]], %[[#D]]) : !cir.double
// CIR-BASIC-NEXT: %{{.+}} = cir.complex.create %[[#E]], %[[#F]] : !cir.double -> !cir.complex<!cir.double>

//      CIR-BASIC: %[[#LHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-BASIC-NEXT: %[[#LHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-BASIC-NEXT: %[[#RHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-BASIC-NEXT: %[[#RHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-BASIC-NEXT: %[[#A:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSR]]) : !s32i
// CIR-BASIC-NEXT: %[[#B:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSI]]) : !s32i
// CIR-BASIC-NEXT: %[[#C:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSI]]) : !s32i
// CIR-BASIC-NEXT: %[[#D:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSR]]) : !s32i
// CIR-BASIC-NEXT: %[[#E:]] = cir.binop(sub, %[[#A]], %[[#B]]) : !s32i
// CIR-BASIC-NEXT: %[[#F:]] = cir.binop(add, %[[#C]], %[[#D]]) : !s32i
// CIR-BASIC-NEXT: %{{.+}} = cir.complex.create %[[#E]], %[[#F]] : !s32i -> !cir.complex<!s32i>

//      LLVM-BASIC: %[[#LHSR:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-BASIC-NEXT: %[[#LHSI:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-BASIC-NEXT: %[[#RHSR:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-BASIC-NEXT: %[[#RHSI:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-BASIC-NEXT: %[[#A:]] = fmul double %[[#LHSR]], %[[#RHSR]]
// LLVM-BASIC-NEXT: %[[#B:]] = fmul double %[[#LHSI]], %[[#RHSI]]
// LLVM-BASIC-NEXT: %[[#C:]] = fmul double %[[#LHSR]], %[[#RHSI]]
// LLVM-BASIC-NEXT: %[[#D:]] = fmul double %[[#LHSI]], %[[#RHSR]]
// LLVM-BASIC-NEXT: %[[#E:]] = fsub double %[[#A]], %[[#B]]
// LLVM-BASIC-NEXT: %[[#F:]] = fadd double %[[#C]], %[[#D]]
// LLVM-BASIC-NEXT: %[[#G:]] = insertvalue { double, double } undef, double %[[#E]], 0
// LLVM-BASIC-NEXT: %{{.+}} = insertvalue { double, double } %[[#G]], double %[[#F]], 1

//      LLVM-BASIC: %[[#LHSR:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-BASIC-NEXT: %[[#LHSI:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-BASIC-NEXT: %[[#RHSR:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-BASIC-NEXT: %[[#RHSI:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-BASIC-NEXT: %[[#A:]] = mul i32 %[[#LHSR]], %[[#RHSR]]
// LLVM-BASIC-NEXT: %[[#B:]] = mul i32 %[[#LHSI]], %[[#RHSI]]
// LLVM-BASIC-NEXT: %[[#C:]] = mul i32 %[[#LHSR]], %[[#RHSI]]
// LLVM-BASIC-NEXT: %[[#D:]] = mul i32 %[[#LHSI]], %[[#RHSR]]
// LLVM-BASIC-NEXT: %[[#E:]] = sub i32 %[[#A]], %[[#B]]
// LLVM-BASIC-NEXT: %[[#F:]] = add i32 %[[#C]], %[[#D]]
// LLVM-BASIC-NEXT: %[[#G:]] = insertvalue { i32, i32 } undef, i32 %[[#E]], 0
// LLVM-BASIC-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#G]], i32 %[[#F]], 1

// CIRGEN-IMPROVED: %{{.+}} = cir.complex.binop mul %{{.+}}, %{{.+}} range(improved) : !cir.complex<!cir.double>
// CIRGEN-IMPROVED: %{{.+}} = cir.complex.binop mul %{{.+}}, %{{.+}} range(improved) : !cir.complex<!s32i>

//      CIR-IMPROVED: %[[#LHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-IMPROVED-NEXT: %[[#LHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-IMPROVED-NEXT: %[[#RHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-IMPROVED-NEXT: %[[#RHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-IMPROVED-NEXT: %[[#A:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSR]]) : !cir.double
// CIR-IMPROVED-NEXT: %[[#B:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSI]]) : !cir.double
// CIR-IMPROVED-NEXT: %[[#C:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSI]]) : !cir.double
// CIR-IMPROVED-NEXT: %[[#D:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSR]]) : !cir.double
// CIR-IMPROVED-NEXT: %[[#E:]] = cir.binop(sub, %[[#A]], %[[#B]]) : !cir.double
// CIR-IMPROVED-NEXT: %[[#F:]] = cir.binop(add, %[[#C]], %[[#D]]) : !cir.double
// CIR-IMPROVED-NEXT: %{{.+}} = cir.complex.create %[[#E]], %[[#F]] : !cir.double -> !cir.complex<!cir.double>

//      CIR-IMPROVED: %[[#LHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-IMPROVED-NEXT: %[[#LHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-IMPROVED-NEXT: %[[#RHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-IMPROVED-NEXT: %[[#RHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-IMPROVED-NEXT: %[[#A:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSR]]) : !s32i
// CIR-IMPROVED-NEXT: %[[#B:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSI]]) : !s32i
// CIR-IMPROVED-NEXT: %[[#C:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSI]]) : !s32i
// CIR-IMPROVED-NEXT: %[[#D:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSR]]) : !s32i
// CIR-IMPROVED-NEXT: %[[#E:]] = cir.binop(sub, %[[#A]], %[[#B]]) : !s32i
// CIR-IMPROVED-NEXT: %[[#F:]] = cir.binop(add, %[[#C]], %[[#D]]) : !s32i
// CIR-IMPROVED-NEXT: %{{.+}} = cir.complex.create %[[#E]], %[[#F]] : !s32i -> !cir.complex<!s32i>

//      LLVM-IMPROVED: %[[#LHSR:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-IMPROVED-NEXT: %[[#LHSI:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-IMPROVED-NEXT: %[[#RHSR:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-IMPROVED-NEXT: %[[#RHSI:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-IMPROVED-NEXT: %[[#A:]] = fmul double %[[#LHSR]], %[[#RHSR]]
// LLVM-IMPROVED-NEXT: %[[#B:]] = fmul double %[[#LHSI]], %[[#RHSI]]
// LLVM-IMPROVED-NEXT: %[[#C:]] = fmul double %[[#LHSR]], %[[#RHSI]]
// LLVM-IMPROVED-NEXT: %[[#D:]] = fmul double %[[#LHSI]], %[[#RHSR]]
// LLVM-IMPROVED-NEXT: %[[#E:]] = fsub double %[[#A]], %[[#B]]
// LLVM-IMPROVED-NEXT: %[[#F:]] = fadd double %[[#C]], %[[#D]]
// LLVM-IMPROVED-NEXT: %[[#G:]] = insertvalue { double, double } undef, double %[[#E]], 0
// LLVM-IMPROVED-NEXT: %{{.+}} = insertvalue { double, double } %[[#G]], double %[[#F]], 1

//      LLVM-IMPROVED: %[[#LHSR:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-IMPROVED-NEXT: %[[#LHSI:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-IMPROVED-NEXT: %[[#RHSR:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-IMPROVED-NEXT: %[[#RHSI:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-IMPROVED-NEXT: %[[#A:]] = mul i32 %[[#LHSR]], %[[#RHSR]]
// LLVM-IMPROVED-NEXT: %[[#B:]] = mul i32 %[[#LHSI]], %[[#RHSI]]
// LLVM-IMPROVED-NEXT: %[[#C:]] = mul i32 %[[#LHSR]], %[[#RHSI]]
// LLVM-IMPROVED-NEXT: %[[#D:]] = mul i32 %[[#LHSI]], %[[#RHSR]]
// LLVM-IMPROVED-NEXT: %[[#E:]] = sub i32 %[[#A]], %[[#B]]
// LLVM-IMPROVED-NEXT: %[[#F:]] = add i32 %[[#C]], %[[#D]]
// LLVM-IMPROVED-NEXT: %[[#G:]] = insertvalue { i32, i32 } undef, i32 %[[#E]], 0
// LLVM-IMPROVED-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#G]], i32 %[[#F]], 1

// CIRGEN-FULL: %{{.+}} = cir.complex.binop mul %{{.+}}, %{{.+}} range(full) : !cir.complex<!cir.double>
// CIRGEN-FULL: %{{.+}} = cir.complex.binop mul %{{.+}}, %{{.+}} range(full) : !cir.complex<!s32i>

//      CIR-FULL: %[[#LHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-FULL-NEXT: %[[#LHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-FULL-NEXT: %[[#RHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-FULL-NEXT: %[[#RHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-FULL-NEXT: %[[#A:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSR]]) : !cir.double
// CIR-FULL-NEXT: %[[#B:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSI]]) : !cir.double
// CIR-FULL-NEXT: %[[#C:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSI]]) : !cir.double
// CIR-FULL-NEXT: %[[#D:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSR]]) : !cir.double
// CIR-FULL-NEXT: %[[#E:]] = cir.binop(sub, %[[#A]], %[[#B]]) : !cir.double
// CIR-FULL-NEXT: %[[#F:]] = cir.binop(add, %[[#C]], %[[#D]]) : !cir.double
// CIR-FULL-NEXT: %[[#RES:]] = cir.complex.create %[[#E]], %[[#F]] : !cir.double -> !cir.complex<!cir.double>
// CIR-FULL-NEXT: %[[#COND:]] = cir.cmp(ne, %[[#E]], %[[#E]]) : !cir.double, !cir.bool
// CIR-FULL-NEXT: %{{.+}} = cir.ternary(%[[#COND]], true {
// CIR-FULL-NEXT:   %[[#COND2:]] = cir.cmp(ne, %[[#F]], %[[#F]]) : !cir.double, !cir.bool
// CIR-FULL-NEXT:   %[[#INNER:]] = cir.ternary(%[[#COND2]], true {
// CIR-FULL-NEXT:     %[[#RES2:]] = cir.call @__muldc3(%[[#LHSR]], %[[#LHSI]], %[[#RHSR]], %[[#RHSI]]) : (!cir.double, !cir.double, !cir.double, !cir.double) -> !cir.complex<!cir.double>
// CIR-FULL-NEXT:     cir.yield %[[#RES2]] : !cir.complex<!cir.double>
// CIR-FULL-NEXT:   }, false {
// CIR-FULL-NEXT:     cir.yield %[[#RES]] : !cir.complex<!cir.double>
// CIR-FULL-NEXT:   }) : (!cir.bool) -> !cir.complex<!cir.double>
// CIR-FULL-NEXT:   cir.yield %[[#INNER]] : !cir.complex<!cir.double>
// CIR-FULL-NEXT: }, false {
// CIR-FULL-NEXT:   cir.yield %[[#RES]] : !cir.complex<!cir.double>
// CIR-FULL-NEXT: }) : (!cir.bool) -> !cir.complex<!cir.double>

//      CIR-FULL: %[[#LHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-FULL-NEXT: %[[#LHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-FULL-NEXT: %[[#RHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-FULL-NEXT: %[[#RHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-FULL-NEXT: %[[#A:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSR]]) : !s32i
// CIR-FULL-NEXT: %[[#B:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSI]]) : !s32i
// CIR-FULL-NEXT: %[[#C:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSI]]) : !s32i
// CIR-FULL-NEXT: %[[#D:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSR]]) : !s32i
// CIR-FULL-NEXT: %[[#E:]] = cir.binop(sub, %[[#A]], %[[#B]]) : !s32i
// CIR-FULL-NEXT: %[[#F:]] = cir.binop(add, %[[#C]], %[[#D]]) : !s32i
// CIR-FULL-NEXT: %{{.+}} = cir.complex.create %[[#E]], %[[#F]] : !s32i -> !cir.complex<!s32i>

//      LLVM-FULL:   %[[#LHSR:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-FULL-NEXT:   %[[#LHSI:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-FULL-NEXT:   %[[#RHSR:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-FULL-NEXT:   %[[#RHSI:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-FULL-NEXT:   %[[#A:]] = fmul double %[[#LHSR]], %[[#RHSR]]
// LLVM-FULL-NEXT:   %[[#B:]] = fmul double %[[#LHSI]], %[[#RHSI]]
// LLVM-FULL-NEXT:   %[[#C:]] = fmul double %[[#LHSR]], %[[#RHSI]]
// LLVM-FULL-NEXT:   %[[#D:]] = fmul double %[[#LHSI]], %[[#RHSR]]
// LLVM-FULL-NEXT:   %[[#E:]] = fsub double %[[#A]], %[[#B]]
// LLVM-FULL-NEXT:   %[[#F:]] = fadd double %[[#C]], %[[#D]]
// LLVM-FULL-NEXT:   %[[#G:]] = insertvalue { double, double } undef, double %[[#E]], 0
// LLVM-FULL-NEXT:   %[[#RES:]] = insertvalue { double, double } %[[#G]], double %[[#F]], 1
// LLVM-FULL-NEXT:   %[[#COND:]] = fcmp une double %[[#E]], %[[#E]]
// LLVM-FULL-NEXT:   br i1 %[[#COND]], label %[[#LA:]], label %[[#LB:]]
//      LLVM-FULL: [[#LA]]:
// LLVM-FULL-NEXT:   %[[#H:]] = fcmp une double %[[#F]], %[[#F]]
// LLVM-FULL-NEXT:   br i1 %[[#H]], label %[[#LC:]], label %[[#LD:]]
//      LLVM-FULL: [[#LC]]:
// LLVM-FULL-NEXT:   %[[#RES2:]] = call { double, double } @__muldc3(double %[[#LHSR]], double %[[#LHSI]], double %[[#RHSR]], double %[[#RHSI]])
// LLVM-FULL-NEXT:   br label %[[#LE:]]
//      LLVM-FULL: [[#LD]]:
// LLVM-FULL-NEXT:   br label %[[#LE]]
//      LLVM-FULL: [[#LE]]:
// LLVM-FULL-NEXT:   %[[#RES3:]] = phi { double, double } [ %[[#RES]], %[[#LD]] ], [ %[[#RES2]], %[[#LC]] ]
// LLVM-FULL-NEXT:   br label %[[#LF:]]
//      LLVM-FULL: [[#LF]]:
// LLVM-FULL-NEXT:   br label %[[#LG:]]
//      LLVM-FULL: [[#LB]]:
// LLVM-FULL-NEXT:   br label %[[#LG]]
//      LLVM-FULL: [[#LG]]:
// LLVM-FULL-NEXT:   %26 = phi { double, double } [ %[[#RES]], %[[#LB]] ], [ %[[#RES3]], %[[#LF]] ]

//      LLVM-FULL: %[[#LHSR:]] = extractvalue { i32, i32 } %28, 0
// LLVM-FULL-NEXT: %[[#LHSI:]] = extractvalue { i32, i32 } %28, 1
// LLVM-FULL-NEXT: %[[#RHSR:]] = extractvalue { i32, i32 } %29, 0
// LLVM-FULL-NEXT: %[[#RHSI:]] = extractvalue { i32, i32 } %29, 1
// LLVM-FULL-NEXT: %[[#A:]] = mul i32 %[[#LHSR]], %[[#RHSR]]
// LLVM-FULL-NEXT: %[[#B:]] = mul i32 %[[#LHSI]], %[[#RHSI]]
// LLVM-FULL-NEXT: %[[#C:]] = mul i32 %[[#LHSR]], %[[#RHSI]]
// LLVM-FULL-NEXT: %[[#D:]] = mul i32 %[[#LHSI]], %[[#RHSR]]
// LLVM-FULL-NEXT: %[[#E:]] = sub i32 %[[#A]], %[[#B]]
// LLVM-FULL-NEXT: %[[#F:]] = add i32 %[[#C]], %[[#D]]
// LLVM-FULL-NEXT: %[[#G:]] = insertvalue { i32, i32 } undef, i32 %[[#E]], 0
// LLVM-FULL-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#G]], i32 %[[#F]], 1

// CHECK: }

void div() {
  cd1 = cd1 / cd2;
  ci1 = ci1 / ci2;
}

// CLANG:   @div
// CPPLANG: @_Z3divv

// CIRGEN-BASIC: %{{.+}} = cir.complex.binop div %{{.+}}, %{{.+}} range(basic) : !cir.complex<!cir.double>
// CIRGEN-BASIC: %{{.+}} = cir.complex.binop div %{{.+}}, %{{.+}} range(basic) : !cir.complex<!s32i>

//      CIR-BASIC: %[[#LHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-BASIC-NEXT: %[[#LHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-BASIC-NEXT: %[[#RHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-BASIC-NEXT: %[[#RHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-BASIC-NEXT: %[[#A:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSR]]) : !cir.double
// CIR-BASIC-NEXT: %[[#B:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSI]]) : !cir.double
// CIR-BASIC-NEXT: %[[#C:]] = cir.binop(mul, %[[#RHSR]], %[[#RHSR]]) : !cir.double
// CIR-BASIC-NEXT: %[[#D:]] = cir.binop(mul, %[[#RHSI]], %[[#RHSI]]) : !cir.double
// CIR-BASIC-NEXT: %[[#E:]] = cir.binop(add, %[[#A]], %[[#B]]) : !cir.double
// CIR-BASIC-NEXT: %[[#F:]] = cir.binop(add, %[[#C]], %[[#D]]) : !cir.double
// CIR-BASIC-NEXT: %[[#G:]] = cir.binop(div, %[[#E]], %[[#F]]) : !cir.double
// CIR-BASIC-NEXT: %[[#H:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSR]]) : !cir.double
// CIR-BASIC-NEXT: %[[#I:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSI]]) : !cir.double
// CIR-BASIC-NEXT: %[[#J:]] = cir.binop(sub, %[[#H]], %[[#I]]) : !cir.double
// CIR-BASIC-NEXT: %[[#K:]] = cir.binop(div, %[[#J]], %[[#F]]) : !cir.double
// CIR-BASIC-NEXT: %{{.+}} = cir.complex.create %[[#G]], %[[#K]] : !cir.double -> !cir.complex<!cir.double>

//      CIR-BASIC: %[[#LHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-BASIC-NEXT: %[[#LHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-BASIC-NEXT: %[[#RHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-BASIC-NEXT: %[[#RHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-BASIC-NEXT: %[[#A:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSR]]) : !s32i
// CIR-BASIC-NEXT: %[[#B:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSI]]) : !s32i
// CIR-BASIC-NEXT: %[[#C:]] = cir.binop(mul, %[[#RHSR]], %[[#RHSR]]) : !s32i
// CIR-BASIC-NEXT: %[[#D:]] = cir.binop(mul, %[[#RHSI]], %[[#RHSI]]) : !s32i
// CIR-BASIC-NEXT: %[[#E:]] = cir.binop(add, %[[#A]], %[[#B]]) : !s32i
// CIR-BASIC-NEXT: %[[#F:]] = cir.binop(add, %[[#C]], %[[#D]]) : !s32i
// CIR-BASIC-NEXT: %[[#G:]] = cir.binop(div, %[[#E]], %[[#F]]) : !s32i
// CIR-BASIC-NEXT: %[[#H:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSR]]) : !s32i
// CIR-BASIC-NEXT: %[[#I:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSI]]) : !s32i
// CIR-BASIC-NEXT: %[[#J:]] = cir.binop(sub, %[[#H]], %[[#I]]) : !s32i
// CIR-BASIC-NEXT: %[[#K:]] = cir.binop(div, %[[#J]], %[[#F]]) : !s32i
// CIR-BASIC-NEXT: %{{.+}} = cir.complex.create %[[#G]], %[[#K]] : !s32i -> !cir.complex<!s32i>

//      LLVM-BASIC: %[[#LHSR:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-BASIC-NEXT: %[[#LHSI:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-BASIC-NEXT: %[[#RHSR:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-BASIC-NEXT: %[[#RHSI:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-BASIC-NEXT: %[[#A:]] = fmul double %[[#LHSR]], %[[#RHSR]]
// LLVM-BASIC-NEXT: %[[#B:]] = fmul double %[[#LHSI]], %[[#RHSI]]
// LLVM-BASIC-NEXT: %[[#C:]] = fmul double %[[#RHSR]], %[[#RHSR]]
// LLVM-BASIC-NEXT: %[[#D:]] = fmul double %[[#RHSI]], %[[#RHSI]]
// LLVM-BASIC-NEXT: %[[#E:]] = fadd double %[[#A]], %[[#B]]
// LLVM-BASIC-NEXT: %[[#F:]] = fadd double %[[#C]], %[[#D]]
// LLVM-BASIC-NEXT: %[[#G:]] = fdiv double %[[#E]], %[[#F]]
// LLVM-BASIC-NEXT: %[[#H:]] = fmul double %[[#LHSI]], %[[#RHSR]]
// LLVM-BASIC-NEXT: %[[#I:]] = fmul double %[[#LHSR]], %[[#RHSI]]
// LLVM-BASIC-NEXT: %[[#J:]] = fsub double %[[#H]], %[[#I]]
// LLVM-BASIC-NEXT: %[[#K:]] = fdiv double %[[#J]], %[[#F]]
// LLVM-BASIC-NEXT: %[[#L:]] = insertvalue { double, double } undef, double %[[#G]], 0
// LLVM-BASIC-NEXT: %{{.+}} = insertvalue { double, double } %[[#L]], double %[[#K]], 1

//      LLVM-BASIC: %[[#LHSR:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-BASIC-NEXT: %[[#LHSI:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-BASIC-NEXT: %[[#RHSR:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-BASIC-NEXT: %[[#RHSI:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-BASIC-NEXT: %[[#A:]] = mul i32 %[[#LHSR]], %[[#RHSR]]
// LLVM-BASIC-NEXT: %[[#B:]] = mul i32 %[[#LHSI]], %[[#RHSI]]
// LLVM-BASIC-NEXT: %[[#C:]] = mul i32 %[[#RHSR]], %[[#RHSR]]
// LLVM-BASIC-NEXT: %[[#D:]] = mul i32 %[[#RHSI]], %[[#RHSI]]
// LLVM-BASIC-NEXT: %[[#E:]] = add i32 %[[#A]], %[[#B]]
// LLVM-BASIC-NEXT: %[[#F:]] = add i32 %[[#C]], %[[#D]]
// LLVM-BASIC-NEXT: %[[#G:]] = sdiv i32 %[[#E]], %[[#F]]
// LLVM-BASIC-NEXT: %[[#H:]] = mul i32 %[[#LHSI]], %[[#RHSR]]
// LLVM-BASIC-NEXT: %[[#I:]] = mul i32 %[[#LHSR]], %[[#RHSI]]
// LLVM-BASIC-NEXT: %[[#J:]] = sub i32 %[[#H]], %[[#I]]
// LLVM-BASIC-NEXT: %[[#K:]] = sdiv i32 %[[#J]], %[[#F]]
// LLVM-BASIC-NEXT: %[[#L:]] = insertvalue { i32, i32 } undef, i32 %[[#G]], 0
// LLVM-BASIC-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#L]], i32 %[[#K]], 1

// CIRGEN-IMPROVED: %{{.+}} = cir.complex.binop div %{{.+}}, %{{.+}} range(improved) : !cir.complex<!cir.double>
// CIRGEN-IMPROVED: %{{.+}} = cir.complex.binop div %{{.+}}, %{{.+}} range(improved) : !cir.complex<!s32i>

//      CIR-IMPROVED: %[[#LHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-IMPROVED-NEXT: %[[#LHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-IMPROVED-NEXT: %[[#RHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-IMPROVED-NEXT: %[[#RHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-IMPROVED-NEXT: %[[#A:]] = cir.fabs %[[#RHSR]] : !cir.double
// CIR-IMPROVED-NEXT: %[[#B:]] = cir.fabs %[[#RHSI]] : !cir.double
// CIR-IMPROVED-NEXT: %[[#C:]] = cir.cmp(ge, %[[#A]], %[[#B]]) : !cir.double, !cir.bool
// CIR-IMPROVED-NEXT: %{{.+}} = cir.ternary(%[[#C]], true {
// CIR-IMPROVED-NEXT:   %[[#D:]] = cir.binop(div, %[[#RHSI]], %[[#RHSR]]) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#E:]] = cir.binop(mul, %[[#D]], %[[#RHSI]]) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#F:]] = cir.binop(add, %[[#RHSR]], %[[#E]]) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#G:]] = cir.binop(mul, %[[#LHSI]], %[[#D]]) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#H:]] = cir.binop(add, %[[#LHSR]], %[[#G]]) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#I:]] = cir.binop(div, %[[#H]], %[[#F]]) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#J:]] = cir.binop(mul, %[[#LHSR]], %[[#D]]) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#K:]] = cir.binop(sub, %[[#LHSI]], %[[#J]]) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#L:]] = cir.binop(div, %[[#K]], %[[#F]]) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#M:]] = cir.complex.create %[[#I]], %[[#L]] : !cir.double -> !cir.complex<!cir.double>
// CIR-IMPROVED-NEXT:   cir.yield %[[#M]] : !cir.complex<!cir.double>
// CIR-IMPROVED-NEXT: }, false {
// CIR-IMPROVED-NEXT:   %[[#D:]] = cir.binop(div, %[[#RHSR]], %[[#RHSI]]) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#E:]] = cir.binop(mul, %[[#D]], %[[#RHSR]]) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#F:]] = cir.binop(add, %[[#RHSI]], %[[#E]]) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#G:]] = cir.binop(mul, %[[#LHSR]], %[[#D]]) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#H:]] = cir.binop(add, %[[#G]], %[[#LHSI]]) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#I:]] = cir.binop(div, %[[#H]], %[[#F]]) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#J:]] = cir.binop(mul, %[[#LHSI]], %[[#D]]) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#K:]] = cir.binop(sub, %[[#J]], %4) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#L:]] = cir.binop(div, %[[#K]], %[[#F]]) : !cir.double
// CIR-IMPROVED-NEXT:   %[[#M:]] = cir.complex.create %[[#I]], %[[#L]] : !cir.double -> !cir.complex<!cir.double>
// CIR-IMPROVED-NEXT:   cir.yield %[[#M]] : !cir.complex<!cir.double>
// CIR-IMPROVED-NEXT: }) : (!cir.bool) -> !cir.complex<!cir.double>

//      CIR-IMPROVED: %[[#LHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-IMPROVED-NEXT: %[[#LHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-IMPROVED-NEXT: %[[#RHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-IMPROVED-NEXT: %[[#RHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-IMPROVED-NEXT: %[[#A:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSR]]) : !s32i
// CIR-IMPROVED-NEXT: %[[#B:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSI]]) : !s32i
// CIR-IMPROVED-NEXT: %[[#C:]] = cir.binop(mul, %[[#RHSR]], %[[#RHSR]]) : !s32i
// CIR-IMPROVED-NEXT: %[[#D:]] = cir.binop(mul, %[[#RHSI]], %[[#RHSI]]) : !s32i
// CIR-IMPROVED-NEXT: %[[#E:]] = cir.binop(add, %[[#A]], %[[#B]]) : !s32i
// CIR-IMPROVED-NEXT: %[[#F:]] = cir.binop(add, %[[#C]], %[[#D]]) : !s32i
// CIR-IMPROVED-NEXT: %[[#G:]] = cir.binop(div, %[[#E]], %[[#F]]) : !s32i
// CIR-IMPROVED-NEXT: %[[#H:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSR]]) : !s32i
// CIR-IMPROVED-NEXT: %[[#I:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSI]]) : !s32i
// CIR-IMPROVED-NEXT: %[[#J:]] = cir.binop(sub, %[[#H]], %[[#I]]) : !s32i
// CIR-IMPROVED-NEXT: %[[#K:]] = cir.binop(div, %[[#J]], %[[#F]]) : !s32i
// CIR-IMPROVED-NEXT: %{{.+}} = cir.complex.create %[[#G]], %[[#K]] : !s32i -> !cir.complex<!s32i>

//      LLVM-IMPROVED: %[[#LHSR:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-IMPROVED-NEXT: %[[#LHSI:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-IMPROVED-NEXT: %[[#RHSR:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-IMPROVED-NEXT: %[[#RHSI:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-IMPROVED-NEXT: %[[#A:]] = call double @llvm.fabs.f64(double %[[#RHSR]])
// LLVM-IMPROVED-NEXT: %[[#B:]] = call double @llvm.fabs.f64(double %[[#RHSI]])
// LLVM-IMPROVED-NEXT: %[[#C:]] = fcmp oge double %[[#A]], %[[#B]]
// LLVM-IMPROVED-NEXT: br i1 %[[#C]], label %[[#LA:]], label %[[#LB:]]
//      LLVM-IMPROVED: [[#LA]]:
// LLVM-IMPROVED-NEXT: %[[#D:]] = fdiv double %[[#RHSI]], %[[#RHSR]]
// LLVM-IMPROVED-NEXT: %[[#E:]] = fmul double %[[#D]], %[[#RHSI]]
// LLVM-IMPROVED-NEXT: %[[#F:]] = fadd double %[[#RHSR]], %[[#E]]
// LLVM-IMPROVED-NEXT: %[[#G:]] = fmul double %[[#LHSI]], %[[#D]]
// LLVM-IMPROVED-NEXT: %[[#H:]] = fadd double %[[#LHSR]], %[[#G]]
// LLVM-IMPROVED-NEXT: %[[#I:]] = fdiv double %[[#H]], %[[#F]]
// LLVM-IMPROVED-NEXT: %[[#J:]] = fmul double %[[#LHSR]], %[[#D]]
// LLVM-IMPROVED-NEXT: %[[#K:]] = fsub double %[[#LHSI]], %[[#J]]
// LLVM-IMPROVED-NEXT: %[[#L:]] = fdiv double %[[#K]], %[[#F]]
// LLVM-IMPROVED-NEXT: %[[#M:]] = insertvalue { double, double } undef, double %[[#I]], 0
// LLVM-IMPROVED-NEXT: %[[#N1:]] = insertvalue { double, double } %[[#M]], double %[[#L]], 1
// LLVM-IMPROVED-NEXT: br label %[[#LC:]]
//      LLVM-IMPROVED: [[#LB]]:
// LLVM-IMPROVED-NEXT: %[[#D:]] = fdiv double %[[#RHSR]], %[[#RHSI]]
// LLVM-IMPROVED-NEXT: %[[#E:]] = fmul double %[[#D]], %[[#RHSR]]
// LLVM-IMPROVED-NEXT: %[[#F:]] = fadd double %[[#RHSI]], %[[#E]]
// LLVM-IMPROVED-NEXT: %[[#G:]] = fmul double %[[#LHSR]], %[[#D]]
// LLVM-IMPROVED-NEXT: %[[#H:]] = fadd double %[[#G]], %[[#LHSI]]
// LLVM-IMPROVED-NEXT: %[[#I:]] = fdiv double %[[#H]], %[[#F]]
// LLVM-IMPROVED-NEXT: %[[#J:]] = fmul double %[[#LHSI]], %[[#D]]
// LLVM-IMPROVED-NEXT: %[[#K:]] = fsub double %[[#J]], %[[#LHSR]]
// LLVM-IMPROVED-NEXT: %[[#L:]] = fdiv double %[[#K]], %[[#F]]
// LLVM-IMPROVED-NEXT: %[[#M:]] = insertvalue { double, double } undef, double %[[#I]], 0
// LLVM-IMPROVED-NEXT: %[[#N2:]] = insertvalue { double, double } %[[#M]], double %[[#L]], 1
// LLVM-IMPROVED-NEXT: br label %[[#LC]]
//      LLVM-IMPROVED: [[#LC]]:
// LLVM-IMPROVED-NEXT: %{{.+}} = phi { double, double } [ %[[#N2]], %[[#LB]] ], [ %[[#N1]], %[[#LA]] ]

//      LLVM-IMPROVED: %[[#LHSR:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-IMPROVED-NEXT: %[[#LHSI:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-IMPROVED-NEXT: %[[#RHSR:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-IMPROVED-NEXT: %[[#RHSI:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-IMPROVED-NEXT: %[[#A:]] = mul i32 %[[#LHSR]], %[[#RHSR]]
// LLVM-IMPROVED-NEXT: %[[#B:]] = mul i32 %[[#LHSI]], %[[#RHSI]]
// LLVM-IMPROVED-NEXT: %[[#C:]] = mul i32 %[[#RHSR]], %[[#RHSR]]
// LLVM-IMPROVED-NEXT: %[[#D:]] = mul i32 %[[#RHSI]], %[[#RHSI]]
// LLVM-IMPROVED-NEXT: %[[#E:]] = add i32 %[[#A]], %[[#B]]
// LLVM-IMPROVED-NEXT: %[[#F:]] = add i32 %[[#C]], %[[#D]]
// LLVM-IMPROVED-NEXT: %[[#G:]] = sdiv i32 %[[#E]], %[[#F]]
// LLVM-IMPROVED-NEXT: %[[#H:]] = mul i32 %[[#LHSI]], %[[#RHSR]]
// LLVM-IMPROVED-NEXT: %[[#I:]] = mul i32 %[[#LHSR]], %[[#RHSI]]
// LLVM-IMPROVED-NEXT: %[[#J:]] = sub i32 %[[#H]], %[[#I]]
// LLVM-IMPROVED-NEXT: %[[#K:]] = sdiv i32 %[[#J]], %[[#F]]
// LLVM-IMPROVED-NEXT: %[[#L:]] = insertvalue { i32, i32 } undef, i32 %[[#G]], 0
// LLVM-IMPROVED-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#L]], i32 %[[#K]], 1

// CIRGEN-FULL: %{{.+}} = cir.complex.binop div %{{.+}}, %{{.+}} range(full) : !cir.complex<!cir.double>
// CIRGEN-FULL: %{{.+}} = cir.complex.binop div %{{.+}}, %{{.+}} range(full) : !cir.complex<!s32i>

//      CIR-FULL: %[[#LHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-FULL-NEXT: %[[#LHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-FULL-NEXT: %[[#RHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-FULL-NEXT: %[[#RHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-FULL-NEXT: %{{.+}} = cir.call @__divdc3(%[[#LHSR]], %[[#LHSI]], %[[#RHSR]], %[[#RHSI]]) : (!cir.double, !cir.double, !cir.double, !cir.double) -> !cir.complex<!cir.double>

//      CIR-FULL: %[[#LHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-FULL-NEXT: %[[#LHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-FULL-NEXT: %[[#RHSR:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-FULL-NEXT: %[[#RHSI:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-FULL-NEXT: %[[#A:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSR]]) : !s32i
// CIR-FULL-NEXT: %[[#B:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSI]]) : !s32i
// CIR-FULL-NEXT: %[[#C:]] = cir.binop(mul, %[[#RHSR]], %[[#RHSR]]) : !s32i
// CIR-FULL-NEXT: %[[#D:]] = cir.binop(mul, %[[#RHSI]], %[[#RHSI]]) : !s32i
// CIR-FULL-NEXT: %[[#E:]] = cir.binop(add, %[[#A]], %[[#B]]) : !s32i
// CIR-FULL-NEXT: %[[#F:]] = cir.binop(add, %[[#C]], %[[#D]]) : !s32i
// CIR-FULL-NEXT: %[[#G:]] = cir.binop(div, %[[#E]], %[[#F]]) : !s32i
// CIR-FULL-NEXT: %[[#H:]] = cir.binop(mul, %[[#LHSI]], %[[#RHSR]]) : !s32i
// CIR-FULL-NEXT: %[[#I:]] = cir.binop(mul, %[[#LHSR]], %[[#RHSI]]) : !s32i
// CIR-FULL-NEXT: %[[#J:]] = cir.binop(sub, %[[#H]], %[[#I]]) : !s32i
// CIR-FULL-NEXT: %[[#K:]] = cir.binop(div, %[[#J]], %[[#F]]) : !s32i
// CIR-FULL-NEXT: %{{.+}} = cir.complex.create %[[#G]], %[[#K]] : !s32i -> !cir.complex<!s32i>

//      LLVM-FULL: %[[#LHSR:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-FULL-NEXT: %[[#LHSI:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-FULL-NEXT: %[[#RHSR:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-FULL-NEXT: %[[#RHSI:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-FULL-NEXT: %{{.+}} = call { double, double } @__divdc3(double %[[#LHSR]], double %[[#LHSI]], double %[[#RHSR]], double %[[#RHSI]])

//      LLVM-FULL: %[[#LHSR:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-FULL-NEXT: %[[#LHSI:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-FULL-NEXT: %[[#RHSR:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-FULL-NEXT: %[[#RHSI:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-FULL-NEXT: %[[#A:]] = mul i32 %[[#LHSR]], %[[#RHSR]]
// LLVM-FULL-NEXT: %[[#B:]] = mul i32 %[[#LHSI]], %[[#RHSI]]
// LLVM-FULL-NEXT: %[[#C:]] = mul i32 %[[#RHSR]], %[[#RHSR]]
// LLVM-FULL-NEXT: %[[#D:]] = mul i32 %[[#RHSI]], %[[#RHSI]]
// LLVM-FULL-NEXT: %[[#E:]] = add i32 %[[#A]], %[[#B]]
// LLVM-FULL-NEXT: %[[#F:]] = add i32 %[[#C]], %[[#D]]
// LLVM-FULL-NEXT: %[[#G:]] = sdiv i32 %[[#E]], %[[#F]]
// LLVM-FULL-NEXT: %[[#H:]] = mul i32 %[[#LHSI]], %[[#RHSR]]
// LLVM-FULL-NEXT: %[[#I:]] = mul i32 %[[#LHSR]], %[[#RHSI]]
// LLVM-FULL-NEXT: %[[#J:]] = sub i32 %[[#H]], %[[#I]]
// LLVM-FULL-NEXT: %[[#K:]] = sdiv i32 %[[#J]], %[[#F]]
// LLVM-FULL-NEXT: %[[#L:]] = insertvalue { i32, i32 } undef, i32 %[[#G]], 0
// LLVM-FULL-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#L]], i32 %[[#K]], 1

// CHECK: }

void add_assign() {
  cd1 += cd2;
  ci1 += ci2;
}

// CLANG:   @add_assign
// CPPLANG: @_Z10add_assignv

// CIRGEN: %{{.+}} = cir.binop(add, %{{.+}}, %{{.+}}) : !cir.complex<!cir.double>
// CIRGEN: %{{.+}} = cir.binop(add, %{{.+}}, %{{.+}}) : !cir.complex<!s32i>

// CHECK: }

void sub_assign() {
  cd1 -= cd2;
  ci1 -= ci2;
}

//   CLANG: @sub_assign
// CPPLANG: @_Z10sub_assignv

// CIRGEN: %{{.+}} = cir.binop(sub, %{{.+}}, %{{.+}}) : !cir.complex<!cir.double>
// CIRGEN: %{{.+}} = cir.binop(sub, %{{.+}}, %{{.+}}) : !cir.complex<!s32i>

// CHECK: }

void mul_assign() {
  cd1 *= cd2;
  ci1 *= ci2;
}

//   CLANG: @mul_assign
// CPPLANG: @_Z10mul_assignv

// CIRGEN-BASIC: %{{.+}} = cir.complex.binop mul %{{.+}}, %{{.+}} range(basic) : !cir.complex<!cir.double>
// CIRGEN-BASIC: %{{.+}} = cir.complex.binop mul %{{.+}}, %{{.+}} range(basic) : !cir.complex<!s32i>

// CIRGEN-IMPROVED: %{{.+}} = cir.complex.binop mul %{{.+}}, %{{.+}} range(improved) : !cir.complex<!cir.double>
// CIRGEN-IMPROVED: %{{.+}} = cir.complex.binop mul %{{.+}}, %{{.+}} range(improved) : !cir.complex<!s32i>

// CIRGEN-FULL: %{{.+}} = cir.complex.binop mul %{{.+}}, %{{.+}} range(full) : !cir.complex<!cir.double>
// CIRGEN-FULL: %{{.+}} = cir.complex.binop mul %{{.+}}, %{{.+}} range(full) : !cir.complex<!s32i>

// CHECK: }

void div_assign() {
  cd1 /= cd2;
  ci1 /= ci2;
}

//   CLANG: @div_assign
// CPPLANG: @_Z10div_assignv

// CIRGEN-BASIC: %{{.+}} = cir.complex.binop div %{{.+}}, %{{.+}} range(basic) : !cir.complex<!cir.double>
// CIRGEN-BASIC: %{{.+}} = cir.complex.binop div %{{.+}}, %{{.+}} range(basic) : !cir.complex<!s32i>

// CIRGEN-IMPROVED: %{{.+}} = cir.complex.binop div %{{.+}}, %{{.+}} range(improved) : !cir.complex<!cir.double>
// CIRGEN-IMPROVED: %{{.+}} = cir.complex.binop div %{{.+}}, %{{.+}} range(improved) : !cir.complex<!s32i>

// CIRGEN-FULL: %{{.+}} = cir.complex.binop div %{{.+}}, %{{.+}} range(full) : !cir.complex<!cir.double>
// CIRGEN-FULL: %{{.+}} = cir.complex.binop div %{{.+}}, %{{.+}} range(full) : !cir.complex<!s32i>

// CHECK: }

void unary_plus() {
  cd1 = +cd1;
  ci1 = +ci1;
}

//   CLANG: @unary_plus
// CPPLANG: @_Z10unary_plusv

// CIRGEN: %{{.+}} = cir.unary(plus, %{{.+}}) : !cir.complex<!cir.double>, !cir.complex<!cir.double>
// CIRGEN: %{{.+}} = cir.unary(plus, %{{.+}}) : !cir.complex<!s32i>, !cir.complex<!s32i>

//      CIR: %[[#OPR:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#OPI:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#RESR:]] = cir.unary(plus, %[[#OPR]]) : !cir.double, !cir.double
// CIR-NEXT: %[[#RESI:]] = cir.unary(plus, %[[#OPI]]) : !cir.double, !cir.double
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#RESR]], %[[#RESI]] : !cir.double -> !cir.complex<!cir.double>

//      CIR: %[[#OPR:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#OPI:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#RESR:]] = cir.unary(plus, %[[#OPR]]) : !s32i, !s32i
// CIR-NEXT: %[[#RESI:]] = cir.unary(plus, %[[#OPI]]) : !s32i, !s32i
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#RESR]], %[[#RESI]] : !s32i -> !cir.complex<!s32i>

//      LLVM: %[[#OPR:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-NEXT: %[[#OPI:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-NEXT: %[[#A:]] = insertvalue { double, double } undef, double %[[#OPR]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { double, double } %[[#A]], double %[[#OPI]], 1

//      LLVM: %[[#OPR:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-NEXT: %[[#OPI:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-NEXT: %[[#A:]] = insertvalue { i32, i32 } undef, i32 %[[#OPR]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#A]], i32 %[[#OPI]], 1

// CHECK: }

void unary_minus() {
  cd1 = -cd1;
  ci1 = -ci1;
}

//   CLANG: @unary_minus
// CPPLANG: @_Z11unary_minusv

// CIRGEN: %{{.+}} = cir.unary(minus, %{{.+}}) : !cir.complex<!cir.double>, !cir.complex<!cir.double>
// CIRGEN: %{{.+}} = cir.unary(minus, %{{.+}}) : !cir.complex<!s32i>, !cir.complex<!s32i>

//      CIR: %[[#OPR:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#OPI:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#RESR:]] = cir.unary(minus, %[[#OPR]]) : !cir.double, !cir.double
// CIR-NEXT: %[[#RESI:]] = cir.unary(minus, %[[#OPI]]) : !cir.double, !cir.double
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#RESR]], %[[#RESI]] : !cir.double -> !cir.complex<!cir.double>

//      CIR: %[[#OPR:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#OPI:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#RESR:]] = cir.unary(minus, %[[#OPR]]) : !s32i, !s32i
// CIR-NEXT: %[[#RESI:]] = cir.unary(minus, %[[#OPI]]) : !s32i, !s32i
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#RESR]], %[[#RESI]] : !s32i -> !cir.complex<!s32i>

//      LLVM: %[[#OPR:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-NEXT: %[[#OPI:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-NEXT: %[[#RESR:]] = fneg double %[[#OPR]]
// LLVM-NEXT: %[[#RESI:]] = fneg double %[[#OPI]]
// LLVM-NEXT: %[[#A:]] = insertvalue { double, double } undef, double %[[#RESR]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { double, double } %[[#A]], double %[[#RESI]], 1

//      LLVM: %[[#OPR:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-NEXT: %[[#OPI:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-NEXT: %[[#RESR:]] = sub i32 0, %[[#OPR]]
// LLVM-NEXT: %[[#RESI:]] = sub i32 0, %[[#OPI]]
// LLVM-NEXT: %[[#A:]] = insertvalue { i32, i32 } undef, i32 %[[#RESR]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#A]], i32 %[[#RESI]], 1

// CHECK: }

void unary_not() {
  cd1 = ~cd1;
  ci1 = ~ci1;
}

//   CLANG: @unary_not
// CPPLANG: @_Z9unary_notv

// CIRGEN: %{{.+}} = cir.unary(not, %{{.+}}) : !cir.complex<!cir.double>, !cir.complex<!cir.double>
// CIRGEN: %{{.+}} = cir.unary(not, %{{.+}}) : !cir.complex<!s32i>, !cir.complex<!s32i>

//      CIR: %[[#OPR:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#OPI:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#RESI:]] = cir.unary(minus, %[[#OPI]]) : !cir.double, !cir.double
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#OPR]], %[[#RESI]] : !cir.double -> !cir.complex<!cir.double>

//      CIR: %[[#OPR:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#OPI:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#RESI:]] = cir.unary(minus, %[[#OPI]]) : !s32i, !s32i
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#OPR]], %[[#RESI]] : !s32i -> !cir.complex<!s32i>

//      LLVM: %[[#OPR:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-NEXT: %[[#OPI:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-NEXT: %[[#RESI:]] = fneg double %[[#OPI]]
// LLVM-NEXT: %[[#A:]] = insertvalue { double, double } undef, double %[[#OPR]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { double, double } %[[#A]], double %[[#RESI]], 1

//      LLVM: %[[#OPR:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-NEXT: %[[#OPI:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-NEXT: %[[#RESI:]] = sub i32 0, %[[#OPI]]
// LLVM-NEXT: %[[#A:]] = insertvalue { i32, i32 } undef, i32 %[[#OPR]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#A]], i32 %[[#RESI]], 1

// CHECK: }

void builtin_conj() {
  cd1 = __builtin_conj(cd1);
}

//   CLANG: @builtin_conj
// CPPLANG: @_Z12builtin_conjv

// CIRGEN: %{{.+}} = cir.unary(not, %{{.+}}) : !cir.complex<!cir.double>, !cir.complex<!cir.double>

//      CIR: %[[#OPR:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#OPI:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#RESI:]] = cir.unary(minus, %[[#OPI]]) : !cir.double, !cir.double
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#OPR]], %[[#RESI]] : !cir.double -> !cir.complex<!cir.double>

//      LLVM: %[[#OPR:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-NEXT: %[[#OPI:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-NEXT: %[[#RESI:]] = fneg double %[[#OPI]]
// LLVM-NEXT: %[[#A:]] = insertvalue { double, double } undef, double %[[#OPR]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { double, double } %[[#A]], double %[[#RESI]], 1

// CHECK: }

void pre_increment() {
  ++cd1;
  ++ci1;
}

//   CLANG: @pre_increment
// CPPLANG: @_Z13pre_incrementv

// CIRGEN: %{{.+}} = cir.unary(inc, %{{.+}}) : !cir.complex<!cir.double>, !cir.complex<!cir.double>
// CIRGEN: %{{.+}} = cir.unary(inc, %{{.+}}) : !cir.complex<!s32i>, !cir.complex<!s32i>

//      CIR: %[[#R:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#I:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#IR:]] = cir.unary(inc, %[[#R]]) : !cir.double, !cir.double
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#IR]], %[[#I]] : !cir.double -> !cir.complex<!cir.double>

//      CIR: %[[#R:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#I:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#IR:]] = cir.unary(inc, %[[#R]]) : !s32i, !s32i
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#IR]], %[[#I]] : !s32i -> !cir.complex<!s32i>

//      LLVM: %[[#R:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-NEXT: %[[#I:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-NEXT: %[[#IR:]] = fadd double 1.000000e+00, %[[#R]]
// LLVM-NEXT: %[[#A:]] = insertvalue { double, double } undef, double %[[#IR]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { double, double } %[[#A]], double %[[#I]], 1

//      LLVM: %[[#R:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-NEXT: %[[#I:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-NEXT: %[[#IR:]] = add i32 %[[#R]], 1
// LLVM-NEXT: %[[#A:]] = insertvalue { i32, i32 } undef, i32 %[[#IR]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#A]], i32 %[[#I]], 1

// CHECK: }

void post_increment() {
  cd1++;
  ci1++;
}

//   CLANG: @post_increment
// CPPLANG: @_Z14post_incrementv

// CIRGEN: %{{.+}} = cir.unary(inc, %{{.+}}) : !cir.complex<!cir.double>, !cir.complex<!cir.double>
// CIRGEN: %{{.+}} = cir.unary(inc, %{{.+}}) : !cir.complex<!s32i>, !cir.complex<!s32i>

//      CIR: %[[#R:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#I:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#IR:]] = cir.unary(inc, %[[#R]]) : !cir.double, !cir.double
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#IR]], %[[#I]] : !cir.double -> !cir.complex<!cir.double>

//      CIR: %[[#R:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#I:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#IR:]] = cir.unary(inc, %[[#R]]) : !s32i, !s32i
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#IR]], %[[#I]] : !s32i -> !cir.complex<!s32i>

//      LLVM: %[[#R:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-NEXT: %[[#I:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-NEXT: %[[#IR:]] = fadd double 1.000000e+00, %[[#R]]
// LLVM-NEXT: %[[#A:]] = insertvalue { double, double } undef, double %[[#IR]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { double, double } %[[#A]], double %[[#I]], 1

//      LLVM: %[[#R:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-NEXT: %[[#I:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-NEXT: %[[#IR:]] = add i32 %[[#R]], 1
// LLVM-NEXT: %[[#A:]] = insertvalue { i32, i32 } undef, i32 %[[#IR]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#A]], i32 %[[#I]], 1

// CHECK: }

void pre_decrement() {
  --cd1;
  --ci1;
}

//   CLANG: @pre_decrement
// CPPLANG: @_Z13pre_decrementv

// CIRGEN: %{{.+}} = cir.unary(dec, %{{.+}}) : !cir.complex<!cir.double>, !cir.complex<!cir.double>
// CIRGEN: %{{.+}} = cir.unary(dec, %{{.+}}) : !cir.complex<!s32i>, !cir.complex<!s32i>

//      CIR: %[[#R:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#I:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#IR:]] = cir.unary(dec, %[[#R]]) : !cir.double, !cir.double
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#IR]], %[[#I]] : !cir.double -> !cir.complex<!cir.double>

//      CIR: %[[#R:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#I:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#IR:]] = cir.unary(dec, %[[#R]]) : !s32i, !s32i
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#IR]], %[[#I]] : !s32i -> !cir.complex<!s32i>

//      LLVM: %[[#R:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-NEXT: %[[#I:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-NEXT: %[[#IR:]] = fadd double -1.000000e+00, %[[#R]]
// LLVM-NEXT: %[[#A:]] = insertvalue { double, double } undef, double %[[#IR]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { double, double } %[[#A]], double %[[#I]], 1

//      LLVM: %[[#R:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-NEXT: %[[#I:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-NEXT: %[[#IR:]] = sub i32 %[[#R]], 1
// LLVM-NEXT: %[[#A:]] = insertvalue { i32, i32 } undef, i32 %[[#IR]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#A]], i32 %[[#I]], 1

// CHECK: }

void post_decrement() {
  cd1--;
  ci1--;
}

//   CLANG: @post_decrement
// CPPLANG: @_Z14post_decrementv

// CIRGEN: %{{.+}} = cir.unary(dec, %{{.+}}) : !cir.complex<!cir.double>, !cir.complex<!cir.double>
// CIRGEN: %{{.+}} = cir.unary(dec, %{{.+}}) : !cir.complex<!s32i>, !cir.complex<!s32i>

//      CIR: %[[#R:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#I:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-NEXT: %[[#IR:]] = cir.unary(dec, %[[#R]]) : !cir.double, !cir.double
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#IR]], %[[#I]] : !cir.double -> !cir.complex<!cir.double>

//      CIR: %[[#R:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#I:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-NEXT: %[[#IR:]] = cir.unary(dec, %[[#R]]) : !s32i, !s32i
// CIR-NEXT: %{{.+}} = cir.complex.create %[[#IR]], %[[#I]] : !s32i -> !cir.complex<!s32i>

//      LLVM: %[[#R:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-NEXT: %[[#I:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-NEXT: %[[#IR:]] = fadd double -1.000000e+00, %[[#R]]
// LLVM-NEXT: %[[#A:]] = insertvalue { double, double } undef, double %[[#IR]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { double, double } %[[#A]], double %[[#I]], 1

//      LLVM: %[[#R:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-NEXT: %[[#I:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-NEXT: %[[#IR:]] = sub i32 %[[#R]], 1
// LLVM-NEXT: %[[#A:]] = insertvalue { i32, i32 } undef, i32 %[[#IR]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#A]], i32 %[[#I]], 1

// CHECK: }
