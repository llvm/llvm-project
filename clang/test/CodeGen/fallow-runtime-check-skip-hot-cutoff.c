// RUN: %clang -fallow-runtime-check-skip-hot-cutoff=1.0 -S -emit-llvm %s -o - -O2 | FileCheck --check-prefix=ONE %s
// RUN: %clang -fallow-runtime-check-skip-hot-cutoff=0.0 -S -emit-llvm %s -o - -O2 | FileCheck --check-prefix=ZERO %s
// RUN: not %clang -fallow-runtime-check-skip-hot-cutoff=6.0 -S -emit-llvm %s -o - -O2 2>&1 | FileCheck --check-prefix=SIX %s
// RUN: not %clang -fallow-runtime-check-skip-hot-cutoff=-1.0 -S -emit-llvm %s -o - -O2 2>&1 | FileCheck --check-prefix=MINUSONE %s
// RUN: not %clang -fallow-runtime-check-skip-hot-cutoff=string -S -emit-llvm %s -o - -O2 2>&1 | FileCheck --check-prefix=STRING %s

// ONE: ret i32 0
// ZERO: ret i32 1
// SIX: invalid value '6.0' in '-fallow-runtime-check-skip-hot-cutoff='
// MINUSONE: invalid value '-1.0' in '-fallow-runtime-check-skip-hot-cutoff='
// STRING: invalid value 'string' in '-fallow-runtime-check-skip-hot-cutoff='

int main(int argc, char** argv) {
    return __builtin_allow_runtime_check("foo");
}
