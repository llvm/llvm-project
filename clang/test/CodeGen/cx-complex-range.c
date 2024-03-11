// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -o - | FileCheck %s --check-prefix=FULL

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=basic -o - | FileCheck %s --check-prefix=BASIC

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -fno-cx-limited-range -o - | FileCheck %s --check-prefix=FULL

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

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -fno-cx-fortran-rules -o - | FileCheck %s --check-prefix=FULL

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN: -ffast-math -complex-range=improved -emit-llvm -o - %s \
// RUN: | FileCheck %s --check-prefix=IMPRVD

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN: -ffast-math -complex-range=promoted -emit-llvm -o - %s \
// RUN: | FileCheck %s --check-prefix=PRMTD

_Complex float divf(_Complex float a, _Complex float b) {
  // LABEL: define {{.*}} @divf(
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

_Complex float mulf(_Complex float a, _Complex float b) {
  // LABEL: define {{.*}} @mulf(
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

_Complex double divd(_Complex double a, _Complex double b) {
  // LABEL: define {{.*}} @divd(
  // FULL: call {{.*}} @__divdc3
  // FULL_FAST: call {{.*}} @__divdc3
  //
  // BASIC: fmul{{.*}}double
  // BASIC-NEXT: fmul{{.*}}double
  // BASIC-NEXT: fadd{{.*}}double
  // BASIC-NEXT: fmul{{.*}}double
  // BASIC-NEXT: fmul{{.*}}double
  // BASIC-NEXT: fadd{{.*}}double
  // BASIC-NEXT: fmul{{.*}}double
  // BASIC-NEXT: fmul{{.*}}double
  // BASIC-NEXT: fsub{{.*}}double
  // BASIC-NEXT: fdiv{{.*}}double
  // BASIC-NEXT: fdiv{{.*}}double
  //
  // IMPRVD: call{{.*}}double @llvm.fabs.f64(double {{.*}})
  // IMPRVD-NEXT: call{{.*}}double @llvm.fabs.f64(double {{.*}})
  // IMPRVD-NEXT: fcmp{{.*}}ugt double {{.*}}, {{.*}}
  // IMPRVD-NEXT:   br i1 {{.*}}, label
  // IMPRVD:  abs_rhsr_greater_or_equal_abs_rhsi:
  // IMPRVD-NEXT: fdiv{{.*}}double
  // IMPRVD-NEXT: fmul{{.*}}double
  // IMPRVD-NEXT: fadd{{.*}}double
  // IMPRVD-NEXT: fmul{{.*}}double
  // IMPRVD-NEXT: fadd{{.*}}double
  // IMPRVD-NEXT: fdiv{{.*}}double
  // IMPRVD-NEXT: fmul{{.*}}double
  // IMPRVD-NEXT: fsub{{.*}}double
  // IMPRVD-NEXT: fdiv{{.*}}double
  // IMPRVD-NEXT: br label
  // IMPRVD: abs_rhsr_less_than_abs_rhsi:
  // IMPRVD-NEXT: fdiv{{.*}}double
  // IMPRVD-NEXT: fmul{{.*}}double
  // IMPRVD-NEXT: fadd{{.*}}double
  // IMPRVD-NEXT: fmul{{.*}}double
  // IMPRVD-NEXT: fadd{{.*}}double
  // IMPRVD-NEXT: fdiv{{.*}}double
  // IMPRVD-NEXT: fmul{{.*}}double
  // IMPRVD-NEXT: fsub{{.*}}double
  // IMPRVD-NEXT: fdiv{{.*}}double
  //
  // PRMTD: load double, ptr {{.*}}
  // PRMTD: fpext double {{.*}} to x86_fp80
  // PRMTD-NEXT: fpext double {{.*}} to x86_fp80
  // PRMTD-NEXT: getelementptr inbounds { double, double }, ptr {{.*}}, i32 0, i32 0
  // PRMTD-NEXT: load double, ptr {{.*}}
  // PRMTD-NEXT: getelementptr inbounds { double, double }, ptr {{.*}}, i32 0, i32 1
  // PRMTD-NEXT: load double, ptr {{.*}}
  // PRMTD-NEXT: fpext double {{.*}} to x86_fp80
  // PRMTD-NEXT: fpext double {{.*}} to x86_fp80
  // PRMTD-NEXT: fmul{{.*}}x86_fp80
  // PRMTD-NEXT: fmul{{.*}}x86_fp80
  // PRMTD-NEXT: fadd{{.*}}x86_fp80
  // PRMTD-NEXT: fmul{{.*}}x86_fp80
  // PRMTD-NEXT: fmul{{.*}}x86_fp80
  // PRMTD-NEXT: fadd{{.*}}x86_fp80
  // PRMTD-NEXT: fmul{{.*}}x86_fp80
  // PRMTD-NEXT: fmul{{.*}}x86_fp80
  // PRMTD-NEXT: fsub{{.*}}x86_fp80
  // PRMTD-NEXT: fdiv{{.*}}x86_fp80
  // PRMTD-NEXT: fdiv{{.*}}x86_fp80
  // PRMTD-NEXT: fptrunc x86_fp80 {{.*}} to double
  // PRMTD-NEXT: fptrunc x86_fp80 {{.*}} to double

  return a / b;
}

_Complex double muld(_Complex double a, _Complex double b) {
  // LABEL: define {{.*}} @muld(
  // FULL: call {{.*}} @__muldc3
  //
  // FULL_FAST: alloca { double, double }
  // FULL_FAST-NEXT: alloca { double, double }
  // FULL_FAST-NEXT: alloca { double, double }
  // FULL_FAST: load double, ptr {{.*}}
  // FULL_FAST-NEXT: getelementptr inbounds { double, double }, ptr {{.*}}, i32 0, i32 1
  // FULL_FAST-NEXT: load double, ptr {{.*}}
  // FULL_FAST-NEXT: getelementptr inbounds { double, double }, ptr {{.*}}, i32 0, i32 0
  // FULL_FAST-NEXT: load double, ptr {{.*}}
  // FULL_FAST-NEXT: getelementptr inbounds { double, double }, ptr {{.*}}, i32 0, i32 1
  // FULL_FAST-NEXT: load double
  // FULL_FAST-NEXT: fmul{{.*}}double
  // FULL_FAST-NEXT: fmul{{.*}}double
  // FULL_FAST-NEXT: fmul{{.*}}double
  // FULL_FAST-NEXT: fmul{{.*}}double
  // FULL_FAST-NEXT: fsub{{.*}}double
  // FULL_FAST-NEXT: fadd{{.*}}double

  // BASIC: alloca { double, double }
  // BASIC-NEXT: alloca { double, double }
  // BASIC-NEXT: alloca { double, double }
  // BASIC: load double, ptr {{.*}}
  // BASIC-NEXT: getelementptr inbounds { double, double }, ptr {{.*}}, i32 0, i32 1
  // BASIC-NEXT: load double, ptr {{.*}}
  // BASIC-NEXT: getelementptr inbounds { double, double }, ptr {{.*}}, i32 0, i32 0
  // BASIC-NEXT: load double, ptr {{.*}}
  // BASIC-NEXT: getelementptr inbounds { double, double }, ptr {{.*}}, i32 0, i32 1
  // BASIC-NEXT: load double
  // BASIC-NEXT: fmul{{.*}}double
  // BASIC-NEXT: fmul{{.*}}double
  // BASIC-NEXT: fmul{{.*}}double
  // BASIC-NEXT: fmul{{.*}}double
  // BASIC-NEXT: fsub{{.*}}double
  // BASIC-NEXT: fadd{{.*}}double
  //
  // IMPRVD: alloca { double, double }
  // IMPRVD-NEXT: alloca { double, double }
  // IMPRVD-NEXT: alloca { double, double }
  // IMPRVD: load double, ptr {{.*}}
  // IMPRVD-NEXT: getelementptr inbounds { double, double }, ptr {{.*}}, i32 0, i32 1
  // IMPRVD-NEXT: load double, ptr {{.*}}
  // IMPRVD-NEXT: getelementptr inbounds { double, double }, ptr {{.*}}, i32 0, i32 0
  // IMPRVD-NEXT: load double, ptr {{.*}}
  // IMPRVD-NEXT: getelementptr inbounds { double, double }, ptr {{.*}}, i32 0, i32 1
  // IMPRVD-NEXT: load double
  // IMPRVD-NEXT: fmul{{.*}}double
  // IMPRVD-NEXT: fmul{{.*}}double
  // IMPRVD-NEXT: fmul{{.*}}double
  // IMPRVD-NEXT: fmul{{.*}}double
  // IMPRVD-NEXT: fsub{{.*}}double
  // IMPRVD-NEXT: fadd{{.*}}double
  //
  // PRMTD: alloca { double, double }
  // PRMTD-NEXT: alloca { double, double }
  // PRMTD-NEXT: alloca { double, double }
  // PRMTD: load double, ptr
  // PRMTD-NEXT: getelementptr inbounds { double, double }, ptr {{.*}}, i32 0, i32 1
  // PRMTD-NEXT: load double, ptr {{.*}}
  // PRMTD-NEXT: getelementptr inbounds { double, double }, ptr {{.*}}, i32 0, i32 0
  // PRMTD-NEXT: load double, ptr {{.*}}
  // PRMTD-NEXT: getelementptr inbounds { double, double }, ptr {{.*}}, i32 0, i32 1
  // PRMTD-NEXT: load{{.*}}double
  // PRMTD-NEXT: fmul{{.*}}double
  // PRMTD-NEXT: fmul{{.*}}double
  // PRMTD-NEXT: fmul{{.*}}double
  // PRMTD-NEXT: fmul{{.*}}double
  // PRMTD-NEXT: fsub{{.*}}double
  // PRMTD-NEXT: fadd{{.*}}double

  return a * b;
}

_Complex _Float16 divf16(_Complex _Float16 a, _Complex _Float16 b) {
  // LABEL: define {{.*}} @divf16(

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
  // PRMTD: load half, ptr {{.*}}
  // PRMTD: fpext half {{.*}} to float
  // PRMTD-NEXT: fpext half {{.*}} to float
  // PRMTD-NEXT: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 0
  // PRMTD-NEXT: load half, ptr {{.*}}
  // PRMTD-NEXT: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 1
  // PRMTD-NEXT: load half, ptr {{.*}}
  // PRMTD-NEXT: fpext half {{.*}} to float
  // PRMTD-NEXT: fpext half {{.*}} to float
  // PRMTD-NEXT: fmul{{.*}}float
  // PRMTD-NEXT: fmul{{.*}}float
  // PRMTD-NEXT: fadd{{.*}}float
  // PRMTD-NEXT: fmul{{.*}}float
  // PRMTD-NEXT: fmul{{.*}}float
  // PRMTD-NEXT: fadd{{.*}}float
  // PRMTD-NEXT: fmul{{.*}}float
  // PRMTD-NEXT: fmul{{.*}}float
  // PRMTD-NEXT: fsub{{.*}}float
  // PRMTD-NEXT: fdiv{{.*}}float
  // PRMTD-NEXT: fdiv{{.*}}float
  // PRMTD-NEXT: fptrunc float {{.*}} to half
  // PRMTD-NEXT: fptrunc float {{.*}} to half

  return a / b;
}

_Complex _Float16 mulf16(_Complex _Float16 a, _Complex _Float16 b) {
  // LABEL: define {{.*}} @mulf16(
  // FULL: call {{.*}} @__mulsc3
  //
  // FULL_FAST: alloca { half, half }
  // FULL_FAST-NEXT: alloca { half, half }
  // FULL_FAST-NEXT: alloca { half, half }
  // FULL_FAST: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 0
  // FULL_FAST-NEXT: load half, ptr {{.*}}
  // FULL_FAST-NEXT: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 1
  // FULL_FAST-NEXT: load half, ptr {{.*}}
  // FULL_FAST-NEXT: fpext half {{.*}} to float
  // FULL_FAST-NEXT: fpext half {{.*}} to float
  // FULL_FAST-NEXT: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 0
  // FULL_FAST-NEXT: load half, ptr {{.*}}
  // FULL_FAST-NEXT: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 1
  // FULL_FAST-NEXT: load half
  // FULL_FAST-NEXT: fpext half {{.*}} to float
  // FULL_FAST-NEXT: fpext half {{.*}} to float
  // FULL_FAST-NEXT: fmul{{.*}}float
  // FULL_FAST-NEXT: fmul{{.*}}float
  // FULL_FAST-NEXT: fmul{{.*}}float
  // FULL_FAST-NEXT: fmul{{.*}}float
  // FULL_FAST-NEXT: fsub{{.*}}float
  // FULL_FAST-NEXT: fadd{{.*}}float
  // FULL_FAST-NEXT: fptrunc float {{.*}} to half
  // FULL_FAST-NEXT: fptrunc float {{.*}} to half

  // BASIC: alloca { half, half }
  // BASIC-NEXT: alloca { half, half }
  // BASIC: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 0
  // BASIC-NEXT: load half, ptr {{.*}}
  // BASIC-NEXT: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 1
  // BASIC-NEXT: load half, ptr {{.*}}
  // BASIC-NEXT: fpext half {{.*}} to float
  // BASIC-NEXT: fpext half {{.*}}  to float
  // BASIC-NEXT: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 0
  // BASIC-NEXT: load half, ptr {{.*}}
  // BASIC-NEXT: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 1
  // BASIC-NEXT: load half
  // BASIC-NEXT: fpext half {{.*}} to float
  // BASIC-NEXT: fpext half {{.*}}  to float
  // BASIC-NEXT: fmul{{.*}}float
  // BASIC-NEXT: fmul{{.*}}float
  // BASIC-NEXT: fmul{{.*}}float
  // BASIC-NEXT: fmul{{.*}}float
  // BASIC-NEXT: fsub{{.*}}float
  // BASIC-NEXT: fadd{{.*}}float
  // BASIC-NEXT: fptrunc float {{.*}} to half
  // BASIC-NEXT: fptrunc float {{.*}} to half
  //
  // IMPRVD: alloca { half, half }
  // IMPRVD-NEXT: alloca { half, half }
  // IMPRVD-NEXT: alloca { half, half }
  // IMPRVD: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 0
  // IMPRVD-NEXT: load half, ptr {{.*}}
  // IMPRVD-NEXT: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 1
  // IMPRVD-NEXT: load half, ptr {{.*}}
  // IMPRVD-NEXT: fpext half {{.*}} to float
  // IMPRVD-NEXT: fpext half {{.*}} to float
  // IMPRVD-NEXT: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 0
  // IMPRVD-NEXT: load half, ptr {{.*}}
  // IMPRVD-NEXT: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 1
  // IMPRVD-NEXT: load half
  // IMPRVD-NEXT: fpext half {{.*}} to float
  // IMPRVD-NEXT: fpext half {{.*}} to float
  // IMPRVD-NEXT: fmul{{.*}}float
  // IMPRVD-NEXT: fmul{{.*}}float
  // IMPRVD-NEXT: fmul{{.*}}float
  // IMPRVD-NEXT: fmul{{.*}}float
  // IMPRVD-NEXT: fsub{{.*}}float
  // IMPRVD-NEXT: fadd{{.*}}float
  // IMPRVD-NEXT: fptrunc float {{.*}} to half
  // IMPRVD-NEXT: fptrunc float {{.*}} to half

  // PRMTD: alloca { half, half }
  // PRMTD-NEXT: alloca { half, half }
  // PRMTD-NEXT: alloca { half, half }
  // PRMTD: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 0
  // PRMTD-NEXT: load half, ptr
  // PRMTD-NEXT: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 1
  // PRMTD-NEXT: load half, ptr {{.*}}
  // PRMTD-NEXT: fpext half {{.*}} to float
  // PRMTD-NEXT: fpext half {{.*}} to float
  // PRMTD-NEXT: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 0
  // PRMTD-NEXT: load half, ptr {{.*}}
  // PRMTD-NEXT: getelementptr inbounds { half, half }, ptr {{.*}}, i32 0, i32 1
  // PRMTD-NEXT: load{{.*}}half
  // PRMTD-NEXT: fpext half {{.*}} to float
  // PRMTD-NEXT: fpext half {{.*}} to float
  // PRMTD-NEXT: fmul{{.*}}float
  // PRMTD-NEXT: fmul{{.*}}float
  // PRMTD-NEXT: fmul{{.*}}float
  // PRMTD-NEXT: fmul{{.*}}float
  // PRMTD-NEXT: fsub{{.*}}float
  // PRMTD-NEXT: fadd{{.*}}float
  // PRMTD-NEXT: fptrunc float {{.*}} to half
  // PRMTD-NEXT: fptrunc float {{.*}} to half

  return a * b;
}

_Complex long double divld(_Complex long double a, _Complex long double b) {
  // LABEL: define {{.*}} @divld(
  // FULL: call {{.*}} @__divxc3
  // FULL_FAST: call {{.*}} @__divxc3
  //
  // BASIC: fmul{{.*}}x86_fp80
  // BASIC-NEXT: fmul{{.*}}x86_fp80
  // BASIC-NEXT: fadd{{.*}}x86_fp80
  // BASIC-NEXT: fmul{{.*}}x86_fp80
  // BASIC-NEXT: fmul{{.*}}x86_fp80
  // BASIC-NEXT: fadd{{.*}}x86_fp80
  // BASIC-NEXT: fmul{{.*}}x86_fp80
  // BASIC-NEXT: fmul{{.*}}x86_fp80
  // BASIC-NEXT: fsub{{.*}}x86_fp80
  // BASIC-NEXT: fdiv{{.*}}x86_fp80
  // BASIC-NEXT: fdiv{{.*}}x86_fp80
  //
  // IMPRVD: call{{.*}}x86_fp80 @llvm.fabs.f80(x86_fp80 {{.*}})
  // IMPRVD-NEXT: call{{.*}}x86_fp80 @llvm.fabs.f80(x86_fp80 {{.*}})
  // IMPRVD-NEXT: fcmp{{.*}}ugt x86_fp80 {{.*}}, {{.*}}
  // IMPRVD-NEXT:   br i1 {{.*}}, label
  // IMPRVD:  abs_rhsr_greater_or_equal_abs_rhsi:
  // IMPRVD-NEXT: fdiv{{.*}}x86_fp80
  // IMPRVD-NEXT: fmul{{.*}}x86_fp80
  // IMPRVD-NEXT: fadd{{.*}}x86_fp80
  // IMPRVD-NEXT: fmul{{.*}}x86_fp80
  // IMPRVD-NEXT: fadd{{.*}}x86_fp80
  // IMPRVD-NEXT: fdiv{{.*}}x86_fp80
  // IMPRVD-NEXT: fmul{{.*}}x86_fp80
  // IMPRVD-NEXT: fsub{{.*}}x86_fp80
  // IMPRVD-NEXT: fdiv{{.*}}x86_fp80
  // IMPRVD-NEXT: br label
  // IMPRVD: abs_rhsr_less_than_abs_rhsi:
  // IMPRVD-NEXT: fdiv{{.*}}x86_fp80
  // IMPRVD-NEXT: fmul{{.*}}x86_fp80
  // IMPRVD-NEXT: fadd{{.*}}x86_fp80
  // IMPRVD-NEXT: fmul{{.*}}x86_fp80
  // IMPRVD-NEXT: fadd{{.*}}x86_fp80
  // IMPRVD-NEXT: fdiv{{.*}}x86_fp80
  // IMPRVD-NEXT: fmul{{.*}}x86_fp80
  // IMPRVD-NEXT: fsub{{.*}}x86_fp80
  // IMPRVD-NEXT: fdiv{{.*}}x86_fp80
  //
  // PRMTD: alloca { x86_fp80, x86_fp80 }
  // PRMTD-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 0
  // PRMTD-NEXT: load x86_fp80, ptr {{.*}}
  // PRMTD-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 1
  // PRMTD-NEXT: load x86_fp80, ptr {{.*}}
  // PRMTD-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 0
  // PRMTD-NEXT: load x86_fp80, ptr {{.*}}
  // PRMTD-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }
  // PRMTD-NEXT: load x86_fp80, ptr {{.*}}
  // PRMTD-NEXT: call{{.*}}x86_fp80 @llvm.fabs.f80(x86_fp80 {{.*}})
  // PRMTD-NEXT: call{{.*}}x86_fp80 @llvm.fabs.f80(x86_fp80 {{.*}})
  // PRMTD-NEXT: fcmp{{.*}}ugt x86_fp80 {{.*}},{{.*}}
  // PRMTD-NEXT: br i1 {{.*}}, label {{.*}}, label {{.*}}
  // PRMTD: abs_rhsr_greater_or_equal_abs_rhsi:
  // PRMTD-NEXT: fdiv{{.*}} x86_fp80
  // PRMTD-NEXT: fmul{{.*}}x86_fp80
  // PRMTD-NEXT: fadd{{.*}}x86_fp80
  // PRMTD-NEXT: fmul{{.*}}x86_fp80
  // PRMTD-NEXT: fadd{{.*}}x86_fp80
  // PRMTD-NEXT: fdiv{{.*}}x86_fp80
  // PRMTD-NEXT: fmul{{.*}}x86_fp80
  // PRMTD-NEXT: fsub{{.*}}x86_fp80
  // PRMTD-NEXT: fdiv{{.*}}x86_fp80
  // PRMTD-NEXT: br label
  // PRMTD: abs_rhsr_less_than_abs_rhsi:
  // PRMTD-NEXT: fdiv{{.*}}x86_fp80
  // PRMTD-NEXT: fmul{{.*}}x86_fp80
  // PRMTD-NEXT: fadd{{.*}}x86_fp80
  // PRMTD-NEXT: fmul{{.*}}x86_fp80
  // PRMTD-NEXT: fadd{{.*}}x86_fp80
  // PRMTD-NEXT: fdiv{{.*}}x86_fp80
  // PRMTD-NEXT: fmul{{.*}}x86_fp80
  // PRMTD-NEXT: fsub{{.*}}x86_fp80
  // PRMTD-NEXT: fdiv {{.*}}x86_fp80

  return a / b;
}

_Complex long double mulld(_Complex long double a, _Complex long double b) {
  // LABEL: define {{.*}} @mulld(
  // FULL: call {{.*}} @__mulxc3

  // FULL_FAST: alloca { x86_fp80, x86_fp80 }
  // FULL_FAST-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 0
  // FULL_FAST-NEXT: load x86_fp80, ptr {{.*}}
  // FULL_FAST-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 1
  // FULL_FAST-NEXT: load x86_fp80, ptr {{.*}}
  // FULL_FAST-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 0
  // FULL_FAST-NEXT: load x86_fp80
  // FULL_FAST-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 1
  // FULL_FAST-NEXT: load x86_fp80, ptr {{.*}}
  // FULL_FAST-NEXT: fmul{{.*}}x86_fp80
  // FULL_FAST-NEXT: fmul{{.*}}x86_fp80
  // FULL_FAST-NEXT: fmul{{.*}}x86_fp80
  // FULL_FAST-NEXT: fmul{{.*}}x86_fp80
  // FULL_FAST-NEXT: fsub{{.*}}x86_fp80
  // FULL_FAST-NEXT: fadd{{.*}}x86_fp80

  // BASIC: alloca { x86_fp80, x86_fp80 }
  // BASIC-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 0
  // BASIC-NEXT: load x86_fp80, ptr {{.*}}
  // BASIC-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 1
  // BASIC-NEXT: load x86_fp80, ptr {{.*}}
  // BASIC-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 0
  // BASIC-NEXT: load x86_fp80
  // BASIC-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 1
  // BASIC-NEXT: load x86_fp80, ptr {{.*}}
  // BASIC-NEXT: fmul{{.*}}x86_fp80
  // BASIC-NEXT: fmul{{.*}}x86_fp80
  // BASIC-NEXT: fmul{{.*}}x86_fp80
  // BASIC-NEXT: fmul{{.*}}x86_fp80
  // BASIC-NEXT: fsub{{.*}}x86_fp80
  // BASIC-NEXT: fadd{{.*}}x86_fp80
  //
  // IMPRVD: alloca { x86_fp80, x86_fp80 }
  // IMPRVD-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 0
  // IMPRVD-NEXT: load x86_fp80, ptr {{.*}}
  // IMPRVD-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 1
  // IMPRVD-NEXT: load x86_fp80, ptr {{.*}}
  // IMPRVD-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 0
  // IMPRVD-NEXT: load x86_fp80
  // IMPRVD-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 1
  // IMPRVD-NEXT: load x86_fp80, ptr {{.*}}
  // IMPRVD-NEXT: fmul{{.*}}x86_fp80
  // IMPRVD-NEXT: fmul{{.*}}x86_fp80
  // IMPRVD-NEXT: fmul{{.*}}x86_fp80
  // IMPRVD-NEXT: fmul{{.*}}x86_fp80
  // IMPRVD-NEXT: fsub{{.*}}x86_fp80
  // IMPRVD-NEXT: fadd{{.*}}x86_fp80
  //
  // PRMTD: alloca { x86_fp80, x86_fp80 }
  // PRMTD-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 0
  // PRMTD-NEXT: load x86_fp80, ptr {{.*}}
  // PRMTD-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 1
  // PRMTD-NEXT: load x86_fp80, ptr {{.*}}
  // PRMTD-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 0
  // PRMTD-NEXT: load{{.*}}x86_fp80
  // PRMTD-NEXT: getelementptr inbounds { x86_fp80, x86_fp80 }, ptr {{.*}}, i32 0, i32 1
  // PRMTD-NEXT: load x86_fp80, ptr {{.*}}
  // PRMTD-NEXT: fmul{{.*}}x86_fp80
  // PRMTD-NEXT: fmul{{.*}}x86_fp80
  // PRMTD-NEXT: fmul{{.*}}x86_fp80
  // PRMTD-NEXT: fmul{{.*}}x86_fp80
  // PRMTD-NEXT: fsub{{.*}}x86_fp80
  // PRMTD-NEXT: fadd{{.*}}x86_fp80

  return a * b;
}
