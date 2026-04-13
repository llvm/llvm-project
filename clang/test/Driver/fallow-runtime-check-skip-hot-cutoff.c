// RUN: %clang -### -fallow-runtime-check-skip-hot-cutoff=1.0 %s 2>&1 | FileCheck %s
// CHECK: -fallow-runtime-check-skip-hot-cutoff=1.0

int main(int argc, char** argv) {
    return __builtin_allow_runtime_check("foo");
}
