// RUN: %clang_hwasan %s -o %t && %env_hwasan_opts=verbose_threads=1 %run %t 2>&1 | FileCheck %s --implicit-check-not=0x000000000000
// REQUIRES: stable-runtime

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

#include <sanitizer/hwasan_interface.h>

// CHECK: sizeof(Thread): [[#]] sizeof(HeapRB): [[#]] sizeof(StackRB): [[#]]
// CHECK: Creating : T0 0x[[#%x,T0:]] stack: [0x[[#%x,SB0:]],0x[[#%x,SE0:]]) sz: [[#]] tls: [0x[[#%x,TB0:]],0x[[#%x,TE0:]])
// CHECK: Creating : T1 0x[[#%x,T1:]] stack: [0x[[#%x,SB1:]],0x[[#%x,SE1:]]) sz: [[#]] tls: [0x[[#%x,TB1:]],0x[[#%x,TE1:]])
// CHECK: Creating : T2 0x[[#%x,T2:]] stack: [0x[[#%x,SB2:]],0x[[#%x,SE2:]]) sz: [[#]] tls: [0x[[#%x,TB2:]],0x[[#%x,TE2:]])
// CHECK: Destroying: T2 0x{{0*}}[[#T2]] stack: [0x{{0*}}[[#SB2]],0x{{0*}}[[#SE2]]) sz: [[#]] tls: [0x{{0*}}[[#TB2]],0x{{0*}}[[#TE2]])
// CHECK: Destroying: T1 0x{{0*}}[[#T1]] stack: [0x{{0*}}[[#SB1]],0x{{0*}}[[#SE1]]) sz: [[#]] tls: [0x{{0*}}[[#TB1]],0x{{0*}}[[#TE1]])

void *Empty(void *arg) {
  if (arg) return NULL;
  pthread_t t;
  pthread_create(&t, NULL, Empty, &t);
  pthread_join(t, NULL);
  return NULL;
}

int main() {
  pthread_t t;
  pthread_create(&t, NULL, Empty, NULL);
  pthread_join(t, NULL);
  return 0;
}
