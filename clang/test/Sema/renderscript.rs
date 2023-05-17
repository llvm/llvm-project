// RUN: %clang_cc1 -fsyntax-only -verify -x renderscript -D__RENDERSCRIPT__ %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify -x c %s
// REQUIRES: x86-registered-target

#ifndef __RENDERSCRIPT__
// expected-warning@+2 {{'kernel' attribute ignored}}
#endif
void __attribute__((kernel)) kernel(void) {}

#ifndef __RENDERSCRIPT__
// expected-warning@+4 {{'kernel' attribute ignored}}
#else
// expected-warning@+2 {{'kernel' attribute only applies to functions}}
#endif
int __attribute__((kernel)) global;

#ifndef __RENDERSCRIPT__
// expected-error@+2 {{function return value cannot have __fp16 type; did you forget * ?}}
#endif
__fp16 fp16_return(void);

#ifndef __RENDERSCRIPT__
// expected-error@+2 {{parameters cannot have __fp16 type; did you forget * ?}}
#endif
void fp16_arg(__fp16 p);
