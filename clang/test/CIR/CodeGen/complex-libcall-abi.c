// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -complex-range=full -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -complex-range=full -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -complex-range=full -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

// CIR: [[REC_LD:!rec_anon_struct[0-9]*]] = !cir.struct<{data !cir.f80, data !cir.f80}>
// CIR: [[REC_D:!rec_anon_struct[0-9]*]] = !cir.struct<{data !cir.double, data !cir.double}>

float _Complex divf(float _Complex a, float _Complex b) { return a / b; }

// A float pair is SSE-classified, so the helper return coerces to <2 x float>.
// CIR-LABEL: cir.func {{.*}}@divf
// CIR: %[[COERCED:.*]] = cir.call @__divsc3({{.*}}) : (!cir.float, !cir.float, !cir.float, !cir.float) -> !cir.vector<2 x !cir.float>
// CIR: cir.store %[[COERCED]], %[[SLOT:.*]] : !cir.vector<2 x !cir.float>, !cir.ptr<!cir.vector<2 x !cir.float>>
// CIR: %[[SLOT_PTR:.*]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!cir.vector<2 x !cir.float>> -> !cir.ptr<!cir.complex<!cir.float>>
// CIR: cir.load %[[SLOT_PTR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>

// The caller's own signature is coerced the same way on both paths.
// LLVM: define dso_local <2 x float> @divf(<2 x float> noundef %{{.+}}, <2 x float> noundef %{{.+}})

// LLVMCIR: %[[COERCED:.*]] = call <2 x float> @__divsc3(float %{{.+}}, float %{{.+}}, float %{{.+}}, float %{{.+}})
// LLVMCIR: store <2 x float> %[[COERCED]], ptr %[[SLOT:.+]], align 8
// LLVMCIR: load { float, float }, ptr %[[SLOT]], align 4
// OGCG: call <2 x float> @__divsc3(float noundef %{{.+}}, float noundef %{{.+}}, float noundef %{{.+}}, float noundef %{{.+}})

float _Complex mulf(float _Complex a, float _Complex b) { return a * b; }

// CIR-LABEL: cir.func {{.*}}@mulf
// CIR: cir.call @__mulsc3({{.*}}) : (!cir.float, !cir.float, !cir.float, !cir.float) -> !cir.vector<2 x !cir.float>

// LLVM: define dso_local <2 x float> @mulf(<2 x float> noundef %{{.+}}, <2 x float> noundef %{{.+}})
// LLVMCIR: call <2 x float> @__mulsc3(float %{{.+}}, float %{{.+}}, float %{{.+}}, float %{{.+}})
// OGCG: call <2 x float> @__mulsc3(float noundef %{{.+}}, float noundef %{{.+}}, float noundef %{{.+}}, float noundef %{{.+}})

double _Complex divd(double _Complex a, double _Complex b) { return a / b; }

// A double pair spans two eightbytes, so the return stays a two-field record.
// CIR-LABEL: cir.func {{.*}}@divd
// CIR: cir.call @__divdc3({{.*}}) : (!cir.double, !cir.double, !cir.double, !cir.double) -> [[REC_D]]{{[^0-9]}}

// CIR does not carry noundef onto expanded parameters.
// LLVMCIR: define dso_local { double, double } @divd(double %{{.+}}, double %{{.+}}, double %{{.+}}, double %{{.+}})
// OGCG: define dso_local { double, double } @divd(double noundef %{{.+}}, double noundef %{{.+}}, double noundef %{{.+}}, double noundef %{{.+}})

// LLVMCIR: call { double, double } @__divdc3(double %{{.+}}, double %{{.+}}, double %{{.+}}, double %{{.+}})
// OGCG: call { double, double } @__divdc3(double noundef %{{.+}}, double noundef %{{.+}}, double noundef %{{.+}}, double noundef %{{.+}})

double _Complex muld(double _Complex a, double _Complex b) { return a * b; }

// CIR-LABEL: cir.func {{.*}}@muld
// CIR: cir.call @__muldc3({{.*}}) : (!cir.double, !cir.double, !cir.double, !cir.double) -> [[REC_D]]{{[^0-9]}}

// LLVMCIR: call { double, double } @__muldc3(double %{{.+}}, double %{{.+}}, double %{{.+}}, double %{{.+}})
// OGCG: call { double, double } @__muldc3(double noundef %{{.+}}, double noundef %{{.+}}, double noundef %{{.+}}, double noundef %{{.+}})

long double _Complex divld(long double _Complex a, long double _Complex b) {
  return a / b;
}

// A long double pair is x87-classified, so the return stays a two-field record.
// CIR-LABEL: cir.func {{.*}}@divld
// CIR: cir.call @__divxc3({{.*}}) : (!cir.long_double<!cir.f80>, !cir.long_double<!cir.f80>, !cir.long_double<!cir.f80>, !cir.long_double<!cir.f80>) -> [[REC_LD]]{{[^0-9]}}

// An x87 pair is passed indirectly.
// LLVM: define dso_local { x86_fp80, x86_fp80 } @divld(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16 %{{.+}}, ptr noundef byval({ x86_fp80, x86_fp80 }) align 16 %{{.+}})

// LLVMCIR: call { x86_fp80, x86_fp80 } @__divxc3(x86_fp80 %{{.+}}, x86_fp80 %{{.+}}, x86_fp80 %{{.+}}, x86_fp80 %{{.+}})
// OGCG: call { x86_fp80, x86_fp80 } @__divxc3(x86_fp80 noundef %{{.+}}, x86_fp80 noundef %{{.+}}, x86_fp80 noundef %{{.+}}, x86_fp80 noundef %{{.+}})

long double _Complex mulld(long double _Complex a, long double _Complex b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}}@mulld
// CIR: cir.call @__mulxc3({{.*}}) : (!cir.long_double<!cir.f80>, !cir.long_double<!cir.f80>, !cir.long_double<!cir.f80>, !cir.long_double<!cir.f80>) -> [[REC_LD]]{{[^0-9]}}

// LLVMCIR: call { x86_fp80, x86_fp80 } @__mulxc3(x86_fp80 %{{.+}}, x86_fp80 %{{.+}}, x86_fp80 %{{.+}}, x86_fp80 %{{.+}})
// OGCG: call { x86_fp80, x86_fp80 } @__mulxc3(x86_fp80 noundef %{{.+}}, x86_fp80 noundef %{{.+}}, x86_fp80 noundef %{{.+}}, x86_fp80 noundef %{{.+}})
