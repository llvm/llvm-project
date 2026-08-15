// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

extern float _Complex a;
extern float _Complex b;

// A dynamic initializer at namespace scope is still inside the global's
// initializer region when the helper call is coerced, so the coercion slot has
// no enclosing function to be placed in.  It has to land in the region that
// later becomes the initializer function.
float _Complex g = a / b;

// CIR-LABEL: cir.func {{.*}}@__cxx_global_var_init
// CIR: %[[SLOT:.*]] = cir.alloca "coerce"{{.*}} : !cir.ptr<!cir.vector<2 x !cir.float>>
// CIR: %[[COERCED:.*]] = cir.call @__divsc3({{.*}}) : (!cir.float, !cir.float, !cir.float, !cir.float) -> !cir.vector<2 x !cir.float>
// CIR: cir.store %[[COERCED]], %[[SLOT]] : !cir.vector<2 x !cir.float>, !cir.ptr<!cir.vector<2 x !cir.float>>
// CIR: cir.store{{.*}} %{{.+}}, %{{.+}} : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM-LABEL: @__cxx_global_var_init(
// LLVM: %[[SLOT:.+]] = alloca <2 x float>, align 8
// LLVM: %[[COERCED:.*]] = call <2 x float> @__divsc3(float %{{.+}}, float %{{.+}}, float %{{.+}}, float %{{.+}})
// LLVM: store <2 x float> %[[COERCED]], ptr %[[SLOT]], align 8
// LLVM: store { float, float } %{{.+}}, ptr @g, align 4

// OGCG-LABEL: @__cxx_global_var_init(
// OGCG: call noundef <2 x float> @__divsc3(float noundef %{{.+}}, float noundef %{{.+}}, float noundef %{{.+}}, float noundef %{{.+}})
