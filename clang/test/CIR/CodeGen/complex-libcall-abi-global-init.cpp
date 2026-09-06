// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -complex-range=full -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -complex-range=full -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -complex-range=full -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

extern float _Complex a;
extern float _Complex b;

float _Complex g = a / b;

// CIR-LABEL: cir.func {{.*}}@__cxx_global_var_init()
// CIR: %[[SLOT:.*]] = cir.alloca "coerce" align(8) : !cir.ptr<!cir.vector<2 x !cir.float>>
// CIR: %[[G:.*]] = cir.get_global @g : !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[COERCED:.*]] = cir.call @__divsc3({{.*}}) : (!cir.float, !cir.float, !cir.float, !cir.float) -> !cir.vector<2 x !cir.float>
// CIR: cir.store %[[COERCED]], %[[SLOT]] : !cir.vector<2 x !cir.float>, !cir.ptr<!cir.vector<2 x !cir.float>>
// CIR: %[[SLOT_PTR:.*]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!cir.vector<2 x !cir.float>> -> !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[RESULT:.*]] = cir.load %[[SLOT_PTR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[RESULT]], %[[G]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM-LABEL: define internal void @__cxx_global_var_init()
// LLVMCIR: %[[SLOT:.+]] = alloca <2 x float>, align 8
// LLVMCIR: %[[COERCED:.*]] = call <2 x float> @__divsc3(float %{{.+}}, float %{{.+}}, float %{{.+}}, float %{{.+}})
// LLVMCIR: store <2 x float> %[[COERCED]], ptr %[[SLOT]], align 8
// LLVMCIR: %[[RESULT:.+]] = load { float, float }, ptr %[[SLOT]], align 4
// LLVMCIR: store { float, float } %[[RESULT]], ptr @g, align 4

// OGCG: call noundef <2 x float> @__divsc3(float noundef %{{.+}}, float noundef %{{.+}}, float noundef %{{.+}}, float noundef %{{.+}})

float _Complex h = a * b;

// Multiply puts its call inside the NaN-check region, one level further in, so
// the slot has to be hoisted to the initializer function's entry block.

// CIR-LABEL: cir.func {{.*}}@__cxx_global_var_init.1()
// CIR: %[[SLOT:.*]] = cir.alloca "coerce" align(8) : !cir.ptr<!cir.vector<2 x !cir.float>>
// CIR: %[[COERCED:.*]] = cir.call @__mulsc3({{.*}}) : (!cir.float, !cir.float, !cir.float, !cir.float) -> !cir.vector<2 x !cir.float>
// CIR: cir.store %[[COERCED]], %[[SLOT]] : !cir.vector<2 x !cir.float>, !cir.ptr<!cir.vector<2 x !cir.float>>

// LLVM-LABEL: define internal void @__cxx_global_var_init.1()
// LLVMCIR: %[[SLOT:.+]] = alloca <2 x float>, align 8
// LLVMCIR: br i1 %{{.+}}, label %[[THEN:.+]], label %{{.+}}
// LLVMCIR: [[THEN]]:
// LLVMCIR: %[[COERCED:.*]] = call <2 x float> @__mulsc3(float %{{.+}}, float %{{.+}}, float %{{.+}}, float %{{.+}})
// LLVMCIR: store <2 x float> %[[COERCED]], ptr %[[SLOT]], align 8
// LLVMCIR: %[[CALLED:.+]] = load { float, float }, ptr %[[SLOT]], align 4
// LLVMCIR: %[[RESULT:.+]] = phi { float, float } [ %{{.+}}, %{{.+}} ], [ %[[CALLED]], %[[THEN]] ]
// LLVMCIR: store { float, float } %[[RESULT]], ptr @h, align 4

// OGCG: call noundef <2 x float> @__mulsc3(float noundef %{{.+}}, float noundef %{{.+}}, float noundef %{{.+}}, float noundef %{{.+}})
