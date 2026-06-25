// RUN: %clang_cc1 -triple arm-linux-gnueabi -fsyntax-only -Wcast-function-type -Wno-cast-function-type-strict -verify=arm %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fsyntax-only -Wcast-function-type -Wno-cast-function-type-strict -verify=aarch64 %s
// RUN: %clang_cc1 -triple m68k-linux-gnu -fsyntax-only -Wcast-function-type -Wno-cast-function-type-strict -verify=m68k %s

typedef void *(*ptr_ret)(void);
typedef unsigned long (*ul_ret)(void);
typedef int (*ptr_param)(void *);
typedef int (*ul_param)(unsigned long);

ptr_ret pr;
ul_ret ur;
ptr_param pp;
ul_param up;

void test(void) {
  // arm-no-diagnostics
  // aarch64-no-diagnostics
  pr = (ptr_ret)ur; // m68k-warning {{cast from 'ul_ret' (aka 'unsigned long (*)(void)') to 'ptr_ret' (aka 'void *(*)(void)') converts to incompatible function type}}
  ur = (ul_ret)pr; // m68k-warning {{cast from 'ptr_ret' (aka 'void *(*)(void)') to 'ul_ret' (aka 'unsigned long (*)(void)') converts to incompatible function type}}
  pp = (ptr_param)up; // m68k-warning {{cast from 'ul_param' (aka 'int (*)(unsigned long)') to 'ptr_param' (aka 'int (*)(void *)') converts to incompatible function type}}
  up = (ul_param)pp; // m68k-warning {{cast from 'ptr_param' (aka 'int (*)(void *)') to 'ul_param' (aka 'int (*)(unsigned long)') converts to incompatible function type}}
}
