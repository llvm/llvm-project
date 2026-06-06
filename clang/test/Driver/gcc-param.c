// RUN: touch %t.o
// RUN: %clang -Werror -Wno-msvc-not-found --param ssp-buffer-size=1 %t.o -###
