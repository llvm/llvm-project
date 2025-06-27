// RUN: %clang -fallow-runtime-check-skip-hot-cutoff=1.0 -S -emit-llvm %s -o - -O2 | FileCheck --check-prefix=ONE %s
// RUN: %clang -fallow-runtime-check-skip-hot-cutoff=0.0 -S -emit-llvm %s -o - -O2 | FileCheck --check-prefix=ZERO %s

// ONE: ret i32 0
// ZERO: ret i32 1

int main(int argc, char** argv) {
    return __builtin_allow_runtime_check("foo");
}
