// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix

// RUN: rm -f %t.pch
// RUN: %clang_cc1 -fmax-type-align=16 -pic-level 2 -fdeprecated-macro -stack-protector 1 -fblocks -fskip-odr-check-in-gmf -fexceptions -fcxx-exceptions -fgnuc-version=0 -triple=%target_triple -DPCH -fincremental-extensions -emit-pch -x c++-header -o %t.pch %s
// RUN: clang-repl -Xcc -fgnuc-version=0 -Xcc -triple=%target_triple -Xcc -include-pch -Xcc %t.pch '#include "%s"' | FileCheck %s

#ifdef PCH
int f_pch() { return 5; }
#endif // PCH

extern "C" int printf(const char *, ...);
auto r1 = printf("f_pch = %d\n", f_pch());
// CHECK: f_pch = 5
