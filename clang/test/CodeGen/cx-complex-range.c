// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -o - | FileCheck %s --check-prefix=FULL

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=basic -o - | FileCheck %s --check-prefix=BASIC

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=improved -o - | FileCheck %s --check-prefix=IMPRVD

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=promoted -o - | FileCheck %s --check-prefix=PRMTD

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=full -o - | FileCheck %s --check-prefix=FULL

// Fast math
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN: -ffast-math -complex-range=basic -emit-llvm -o - %s \
// RUN: | FileCheck %s --check-prefix=BASIC

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN: -ffast-math -complex-range=full -emit-llvm -o - %s \
// RUN: | FileCheck %s --check-prefix=FULL_FAST

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN: -ffast-math -complex-range=improved -emit-llvm -o - %s \
// RUN: | FileCheck %s --check-prefix=IMPRVD

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN: -ffast-math -complex-range=promoted -emit-llvm -o - %s \
// RUN: | FileCheck %s --check-prefix=PRMTD

_Complex float div(_Complex float a, _Complex float b) {
  // LABEL: define {{.*}} @div(
  // FULL: call {{.*}} @__divsc3
  // FULL_FAST: call {{.*}} @__divsc3
  //
  // BASIC: fmul{{.*}}float
  // BASIC-NEXT: fmul{{.*}}float
  // BASIC-NEXT: fadd{{.*}}float
  // BASIC-NEXT: fmul{{.*}}float
  // BASIC-NEXT: fmul{{.*}}float
  // BASIC-NEXT: fadd{{.*}}float
  // BASIC-NEXT: fmul{{.*}}float
  // BASIC-NEXT: fmul{{.*}}float
  // BASIC-NEXT: fsub{{.*}}float
  // BASIC-NEXT: fdiv{{.*}}float
  // BASIC-NEXT: fdiv{{.*}}float
  //
  // IMPRVD: call{{.*}}float @llvm.fabs.f32(float {{.*}})
  // IMPRVD-NEXT: call{{.*}}float @llvm.fabs.f32(float {{.*}})
  // IMPRVD-NEXT: fcmp{{.*}}ugt float {{.*}}, {{.*}}
  // IMPRVD-NEXT:   br i1 {{.*}}, label
  // IMPRVD:  abs_rhsr_greater_or_equal_abs_rhsi:
  // IMPRVD-NEXT: fdiv{{.*}}float
  // IMPRVD-NEXT: fmul{{.*}}float
  // IMPRVD-NEXT: fadd{{.*}}float
  // IMPRVD-NEXT: fmul{{.*}}float
  // IMPRVD-NEXT: fadd{{.*}}float
  // IMPRVD-NEXT: fdiv{{.*}}float
  // IMPRVD-NEXT: fmul{{.*}}float
  // IMPRVD-NEXT: fsub{{.*}}float
  // IMPRVD-NEXT: fdiv{{.*}}float
  // IMPRVD-NEXT: br label
  // IMPRVD: abs_rhsr_less_than_abs_rhsi:
  // IMPRVD-NEXT: fdiv{{.*}}float
  // IMPRVD-NEXT: fmul{{.*}}float
  // IMPRVD-NEXT: fadd{{.*}}float
  // IMPRVD-NEXT: fmul{{.*}}float
  // IMPRVD-NEXT: fadd{{.*}}float
  // IMPRVD-NEXT: fdiv{{.*}}float
  // IMPRVD-NEXT: fmul{{.*}}float
  // IMPRVD-NEXT: fsub{{.*}}float
  // IMPRVD-NEXT: fdiv{{.*}}float
  //
  // PRMTD: load float, ptr {{.*}}
  // PRMTD: fpext float {{.*}} to double
  // PRMTD-NEXT: fpext float {{.*}} to double
  // PRMTD-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
  // PRMTD-NEXT: load float, ptr {{.*}}
  // PRMTD-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
  // PRMTD-NEXT: load float, ptr {{.*}}
  // PRMTD-NEXT: fpext float {{.*}} to double
  // PRMTD-NEXT: fpext float {{.*}} to double
  // PRMTD-NEXT: fmul{{.*}}double
  // PRMTD-NEXT: fmul{{.*}}double
  // PRMTD-NEXT: fadd{{.*}}double
  // PRMTD-NEXT: fmul{{.*}}double
  // PRMTD-NEXT: fmul{{.*}}double
  // PRMTD-NEXT: fadd{{.*}}double
  // PRMTD-NEXT: fmul{{.*}}double
  // PRMTD-NEXT: fmul{{.*}}double
  // PRMTD-NEXT: fsub{{.*}}double
  // PRMTD-NEXT: fdiv{{.*}}double
  // PRMTD-NEXT: fdiv{{.*}}double
  // PRMTD-NEXT: fptrunc double {{.*}} to float
  // PRMTD-NEXT: fptrunc double {{.*}} to float

  return a / b;
}

_Complex float mul(_Complex float a, _Complex float b) {
  // LABEL: define {{.*}} @mul(
  // FULL: call {{.*}} @__mulsc3
  //
  // FULL_FAST: alloca { float, float }
  // FULL_FAST-NEXT: alloca { float, float }
  // FULL_FAST-NEXT: alloca { float, float }
  // FULL_FAST: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
  // FULL_FAST-NEXT: load float, ptr {{.*}}
  // FULL_FAST-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
  // FULL_FAST-NEXT: load float, ptr {{.*}}
  // FULL_FAST-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
  // FULL_FAST-NEXT: load float, ptr {{.*}}
  // FULL_FAST-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
  // FULL_FAST-NEXT: load float
  // FULL_FAST-NEXT: fmul{{.*}}float
  // FULL_FAST-NEXT: fmul{{.*}}float
  // FULL_FAST-NEXT: fmul{{.*}}float
  // FULL_FAST-NEXT: fmul{{.*}}float
  // FULL_FAST-NEXT: fsub{{.*}}float
  // FULL_FAST-NEXT: fadd{{.*}}float

  // BASIC: alloca { float, float }
  // BASIC-NEXT: alloca { float, float }
  // BASIC-NEXT: alloca { float, float }
  // BASIC: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
  // BASIC-NEXT: load float, ptr {{.*}}
  // BASIC-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
  // BASIC-NEXT: load float, ptr {{.*}}
  // BASIC-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
  // BASIC-NEXT: load float, ptr {{.*}}
  // BASIC-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
  // BASIC-NEXT: load float
  // BASIC-NEXT: fmul{{.*}}float
  // BASIC-NEXT: fmul{{.*}}float
  // BASIC-NEXT: fmul{{.*}}float
  // BASIC-NEXT: fmul{{.*}}float
  // BASIC-NEXT: fsub{{.*}}float
  // BASIC-NEXT: fadd{{.*}}float
  //
  // IMPRVD: alloca { float, float }
  // IMPRVD-NEXT: alloca { float, float }
  // IMPRVD-NEXT: alloca { float, float }
  // IMPRVD: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
  // IMPRVD-NEXT: load float, ptr {{.*}}
  // IMPRVD-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
  // IMPRVD-NEXT: load float, ptr {{.*}}
  // IMPRVD-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
  // IMPRVD-NEXT: load float, ptr {{.*}}
  // IMPRVD-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
  // IMPRVD-NEXT: load float
  // IMPRVD-NEXT: fmul{{.*}}float
  // IMPRVD-NEXT: fmul{{.*}}float
  // IMPRVD-NEXT: fmul{{.*}}float
  // IMPRVD-NEXT: fmul{{.*}}float
  // IMPRVD-NEXT: fsub{{.*}}float
  // IMPRVD-NEXT: fadd{{.*}}float
  //
  // PRMTD: alloca { float, float }
  // PRMTD-NEXT: alloca { float, float }
  // PRMTD-NEXT: alloca { float, float }
  // PRMTD: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
  // PRMTD-NEXT: load float, ptr
  // PRMTD-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
  // PRMTD-NEXT: load float, ptr {{.*}}
  // PRMTD-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
  // PRMTD-NEXT: load float, ptr {{.*}}
  // PRMTD-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
  // PRMTD-NEXT: load{{.*}}float
  // PRMTD-NEXT: fmul{{.*}}float
  // PRMTD-NEXT: fmul{{.*}}float
  // PRMTD-NEXT: fmul{{.*}}float
  // PRMTD-NEXT: fmul{{.*}}float
  // PRMTD-NEXT: fsub{{.*}}float
  // PRMTD-NEXT: fadd{{.*}}float

  return a * b;
}
