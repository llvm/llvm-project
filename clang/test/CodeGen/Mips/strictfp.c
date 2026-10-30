// RUN: %clang_cc1 -triple mips64-unknown-linux-gnu -frounding-math \
// RUN:   -ffp-exception-behavior=strict -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=STRICT
//
// STRICT: call double @llvm.experimental.constrained.fdiv.f64(double 1.000000e+00, double 3.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict")
double div(void) { return 1.0 / 3.0; }

