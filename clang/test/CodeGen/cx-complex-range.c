// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -o - | FileCheck %s --check-prefix=FULL

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=limited -o - | FileCheck %s --check-prefix=LMTD

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=smith -o - | FileCheck %s --check-prefix=SMITH

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=extend -o - | FileCheck %s --check-prefix=EXTND

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=full -o - | FileCheck %s --check-prefix=FULL

// Fast math
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN: -ffast-math -complex-range=limited -emit-llvm -o - %s \
// RUN: | FileCheck %s --check-prefix=LMTD

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN: -ffast-math -complex-range=full -emit-llvm -o - %s \
// RUN: | FileCheck %s --check-prefix=FULL

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN: -ffast-math -complex-range=smith -emit-llvm -o - %s \
// RUN: | FileCheck %s --check-prefix=SMITH

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN: -ffast-math -complex-range=extend -emit-llvm -o - %s \
// RUN: | FileCheck %s --check-prefix=EXTND

_Complex float div(_Complex float a, _Complex float b) {
  // LABEL: define {{.*}} @div(
  // FULL: call {{.*}} @__divsc3
  //
  // LMTD: fmul{{.*}}float
  // LMTD-NEXT: fmul{{.*}}float
  // LMTD-NEXT: fadd{{.*}}float
  // LMTD-NEXT: fmul{{.*}}float
  // LMTD-NEXT: fmul{{.*}}float
  // LMTD-NEXT: fadd{{.*}}float
  // LMTD-NEXT: fmul{{.*}}float
  // LMTD-NEXT: fmul{{.*}}float
  // LMTD-NEXT: fsub{{.*}}float
  // LMTD-NEXT: fdiv{{.*}}float
  // LMTD-NEXT: fdiv{{.*}}float
  //
  // SMITH: call{{.*}}float @llvm.fabs.f32(float {{.*}})
  // SMITH-NEXT: call{{.*}}float @llvm.fabs.f32(float {{.*}})
  // SMITH-NEXT: fcmp{{.*}}ugt float {{.*}}, {{.*}}
  // SMITH-NEXT:   br i1 {{.*}}, label
  // SMITH:  abs_rhsr_greater_or_equal_abs_rhsi:
  // SMITH-NEXT: fdiv{{.*}}float
  // SMITH-NEXT: fmul{{.*}}float
  // SMITH-NEXT: fadd{{.*}}float
  // SMITH-NEXT: fmul{{.*}}float
  // SMITH-NEXT: fadd{{.*}}float
  // SMITH-NEXT: fdiv{{.*}}float
  // SMITH-NEXT: fmul{{.*}}float
  // SMITH-NEXT: fsub{{.*}}float
  // SMITH-NEXT: fdiv{{.*}}float
  // SMITH-NEXT: br label
  // SMITH: abs_rhsr_less_than_abs_rhsi:
  // SMITH-NEXT: fdiv{{.*}}float
  // SMITH-NEXT: fmul{{.*}}float
  // SMITH-NEXT: fadd{{.*}}float
  // SMITH-NEXT: fmul{{.*}}float
  // SMITH-NEXT: fadd{{.*}}float
  // SMITH-NEXT: fdiv{{.*}}float
  // SMITH-NEXT: fmul{{.*}}float
  // SMITH-NEXT: fsub{{.*}}float
  // SMITH-NEXT: fdiv{{.*}}float
  //
  // EXTND: load float, ptr {{.*}}
  // EXTND: fpext float {{.*}} to double
  // EXTND-NEXT: fpext float {{.*}} to double
  // EXTND-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
  // EXTND-NEXT: load float, ptr {{.*}}
  // EXTND-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
  // EXTND-NEXT: load float, ptr {{.*}}
  // EXTND-NEXT: fpext float {{.*}} to double
  // EXTND-NEXT: fpext float {{.*}} to double
  // EXTND-NEXT: fmul{{.*}}double
  // EXTND-NEXT: fmul{{.*}}double
  // EXTND-NEXT: fadd{{.*}}double
  // EXTND-NEXT: fmul{{.*}}double
  // EXTND-NEXT: fmul{{.*}}double
  // EXTND-NEXT: fadd{{.*}}double
  // EXTND-NEXT: fmul{{.*}}double
  // EXTND-NEXT: fmul{{.*}}double
  // EXTND-NEXT: fsub{{.*}}double
  // EXTND-NEXT: fdiv{{.*}}double
  // EXTND-NEXT: fdiv{{.*}}double
  // EXTND-NEXT: fptrunc double {{.*}} to float
  // EXTND-NEXT: fptrunc double {{.*}} to float

  return a / b;
}

_Complex float mul(_Complex float a, _Complex float b) {
  // LABEL: define {{.*}} @mul(
  // FULL: call {{.*}} @__mulsc3
  //
  // LMTD: alloca { float, float }
  // LMTD-NEXT: alloca { float, float }
  // LMTD-NEXT: alloca { float, float }
  // LMTD: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
  // LMTD-NEXT: load float, ptr {{.*}}
  // LMTD-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
  // LMTD-NEXT: load float, ptr {{.*}}
  // LMTD-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
  // LMTD-NEXT: load float, ptr {{.*}}
  // LMTD-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
  // LMTD-NEXT: load float
  // LMTD-NEXT: fmul{{.*}}float
  // LMTD-NEXT: fmul{{.*}}float
  // LMTD-NEXT: fmul{{.*}}float
  // LMTD-NEXT: fmul{{.*}}float
  // LMTD-NEXT: fsub{{.*}}float
  // LMTD-NEXT: fadd{{.*}}float
  //
  // SMITH: alloca { float, float }
  // SMITH-NEXT: alloca { float, float }
  // SMITH-NEXT: alloca { float, float }
  // SMITH: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
  // SMITH-NEXT: load float, ptr {{.*}}
  // SMITH-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
  // SMITH-NEXT: load float, ptr {{.*}}
  // SMITH-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
  // SMITH-NEXT: load float, ptr {{.*}}
  // SMITH-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
  // SMITH-NEXT: load float
  // SMITH-NEXT: fmul{{.*}}float
  // SMITH-NEXT: fmul{{.*}}float
  // SMITH-NEXT: fmul{{.*}}float
  // SMITH-NEXT: fmul{{.*}}float
  // SMITH-NEXT: fsub{{.*}}float
  // SMITH-NEXT: fadd{{.*}}float
  //
  // EXTND: alloca { float, float }
  // EXTND-NEXT: alloca { float, float }
  // EXTND-NEXT: alloca { float, float }
  // EXTND: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
  // EXTND-NEXT: load float, ptr
  // EXTND-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
  // EXTND-NEXT: load float, ptr {{.*}}
  // EXTND-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
  // EXTND-NEXT: load float, ptr {{.*}}
  // EXTND-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
  // EXTND-NEXT: load{{.*}}float
  // EXTND-NEXT: fmul{{.*}}float
  // EXTND-NEXT: fmul{{.*}}float
  // EXTND-NEXT: fmul{{.*}}float
  // EXTND-NEXT: fmul{{.*}}float
  // EXTND-NEXT: fsub{{.*}}float
  // EXTND-NEXT: fadd{{.*}}float

  return a * b;
}
