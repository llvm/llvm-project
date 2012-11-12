// RUN: %clang_cc1 %s -o /dev/null -emit-llvm

int foo(__complex float c) {
    return creal(c);
}
