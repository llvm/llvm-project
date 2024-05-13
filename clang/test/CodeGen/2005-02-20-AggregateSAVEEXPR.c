// RUN: %clang_cc1 %s -o /dev/null -emit-llvm

double creal(_Complex double);

int foo(__complex float c) {
    return creal(c);
}
