// RUN: not %clang_cc1 -triple powerpc64le-linux-gnu -emit-llvm %s -o -

long __attribute__((target("power8-vector,no-vsx"))) foo (void) { return 0; }  // expected-error {{option '-mpower8-vector' cannot be specified with '-mno-vsx'}}
long __attribute__((target("no-altivec,vsx"))) foo2(void) { return 0; }        // expected-error {{option '-mvsx' cannot be specified with '-mno-altivec'}}
long __attribute__((target("no-hard-float,altivec"))) foo3(void) { return 0; } // expected-error {{option '-msoft-float' cannot be specified with '-maltivec'}}
long __attribute__((target("no-hard-float,vsx"))) foo4(void) { return 0; } // expected-error {{option '-msoft-float' cannot be specified with '-mvsx'}}

