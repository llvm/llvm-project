// RUN: %clang -O1 %s -S -c -emit-llvm -o - | FileCheck %s
// RUN: %clang -O1 %s -S -c -emit-llvm -o - | opt -ipconstprop -S | FileCheck --check-prefix=IPCP %s

// CHECK: declare !callback ![[cid:[0-9]+]] dso_local i32 @pthread_create
// CHECK: ![[cid]] = !{![[cidb:[0-9]+]]}
// CHECK: ![[cidb]] = !{i64 2, i64 3, i1 false}

#include <pthread.h>

const int GlobalVar = 0;

static void *callee0(void *payload) {
// IPCP:      define internal i8* @callee0
// IPCP-NEXT:   entry:
// IPCP-NEXT:     ret i8* null
  return payload;
}

static void *callee1(void *payload) {
// IPCP:      define internal i8* @callee1
// IPCP-NEXT:   entry:
// IPCP-NEXT:     ret i8* bitcast (i32* @GlobalVar to i8*)
  return payload;
}

void foo() {
  pthread_t MyFirstThread;
  pthread_create(&MyFirstThread, NULL, callee0, NULL);

  pthread_t MySecondThread;
  pthread_create(&MySecondThread, NULL, callee1, (void *)&GlobalVar);
}
