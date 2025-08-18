// RUN: %clang_cc1 %s -fopenacc -verify

void *a = ^ { static int b };
