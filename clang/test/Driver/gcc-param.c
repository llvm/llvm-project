// RUN: touch %t.o
// RUN: %clang -Werror --param ssp-buffer-size=1 %t.o -###
