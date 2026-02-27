// Check that indirect call hash tables properly register multiple calls,
// and that calls from different processes don't get mixed up when using
// --instrumentation-file-append-pid.

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__attribute__((noinline)) void funcA(int pid) { printf("funcA %d\n", pid); }
__attribute__((noinline)) void funcB(int pid) { printf("funcB %d\n", pid); }
__attribute__((noinline)) void funcC(int pid) { printf("funcC %d\n", pid); }
__attribute__((noinline)) void funcD(int pid) { printf("funcD %d\n", pid); }
__attribute__((noinline)) void funcE(int pid) { printf("funcE %d\n", pid); }
__attribute__((noinline)) void funcF(int pid) { printf("funcF %d\n", pid); }
__attribute__((noinline)) void funcG(int pid) { printf("funcG %d\n", pid); }
__attribute__((noinline)) void funcH(int pid) { printf("funcH %d\n", pid); }
__attribute__((noinline)) void funcI(int pid) { printf("funcI %d\n", pid); }
__attribute__((noinline)) void funcJ(int pid) { printf("funcJ %d\n", pid); }
__attribute__((noinline)) void funcK(int pid) { printf("funcK %d\n", pid); }
__attribute__((noinline)) void funcL(int pid) { printf("funcL %d\n", pid); }
__attribute__((noinline)) void funcM(int pid) { printf("funcM %d\n", pid); }
__attribute__((noinline)) void funcN(int pid) { printf("funcN %d\n", pid); }
__attribute__((noinline)) void funcO(int pid) { printf("funcO %d\n", pid); }
__attribute__((noinline)) void funcP(int pid) { printf("funcP %d\n", pid); }

int main() {

  void (*funcs[])(int) = {funcA, funcB, funcC, funcD, funcE, funcF,
                          funcG, funcH, funcI, funcJ, funcK, funcL,
                          funcM, funcN, funcO, funcP};
  int i;

  switch (fork()) {
  case -1:
    printf("Failed to fork!\n");
    exit(-1);
    break;
  case 0:
    i = 0;
    break;
  default:
    i = 1;
    break;
  }
  int pid = getpid();
  for (; i < sizeof(funcs) / sizeof(void *); i += 2) {
    funcs[i](pid);
  }

  return 0;
}
/*
REQUIRES: system-linux,fuser

RUN: %clang %cflags %s -o %t.exe -Wl,-q -pie -fpie

RUN: llvm-bolt %t.exe --instrument --instrumentation-file=%t.fdata \
RUN:   --conservative-instrumentation -o %t.instrumented_conservative \
RUN: --instrumentation-sleep-time=1 --instrumentation-no-counters-clear \
RUN: --instrumentation-wait-forks

# Instrumented program needs to finish returning zero
# Both output and profile must contain all 16 functions
# We need to use bash to invoke this as otherwise we hang inside a
# popen.communicate call in lit's internal shell. Eventually we should not
# need this.
# TODO(boomanaiden154): Remove once
# https://github.com/llvm/llvm-project/issues/156484 is fixed.
RUN: bash -c "%t.instrumented_conservative; wait" > %t.output
# We can just read because we ensure the profile will be fully written by
# calling wait inside the bash invocation.
RUN: cat %t.output | FileCheck %s --check-prefix=CHECK-OUTPUT
RUN: cat %t.fdata | FileCheck %s --check-prefix=CHECK-COMMON-PROF

CHECK-OUTPUT-DAG: funcA
CHECK-OUTPUT-DAG: funcB
CHECK-OUTPUT-DAG: funcC
CHECK-OUTPUT-DAG: funcD
CHECK-OUTPUT-DAG: funcE
CHECK-OUTPUT-DAG: funcF
CHECK-OUTPUT-DAG: funcG
CHECK-OUTPUT-DAG: funcH
CHECK-OUTPUT-DAG: funcI
CHECK-OUTPUT-DAG: funcJ
CHECK-OUTPUT-DAG: funcK
CHECK-OUTPUT-DAG: funcL
CHECK-OUTPUT-DAG: funcM
CHECK-OUTPUT-DAG: funcN
CHECK-OUTPUT-DAG: funcO
CHECK-OUTPUT-DAG: funcP

CHECK-COMMON-PROF-DAG: 1 main {{[0-9a-f]+}} 1 funcA 0 0 1
CHECK-COMMON-PROF-DAG: 1 main {{[0-9a-f]+}} 1 funcB 0 0 1
CHECK-COMMON-PROF-DAG: 1 main {{[0-9a-f]+}} 1 funcC 0 0 1
CHECK-COMMON-PROF-DAG: 1 main {{[0-9a-f]+}} 1 funcD 0 0 1
CHECK-COMMON-PROF-DAG: 1 main {{[0-9a-f]+}} 1 funcE 0 0 1
CHECK-COMMON-PROF-DAG: 1 main {{[0-9a-f]+}} 1 funcF 0 0 1
CHECK-COMMON-PROF-DAG: 1 main {{[0-9a-f]+}} 1 funcG 0 0 1
CHECK-COMMON-PROF-DAG: 1 main {{[0-9a-f]+}} 1 funcH 0 0 1
CHECK-COMMON-PROF-DAG: 1 main {{[0-9a-f]+}} 1 funcI 0 0 1
CHECK-COMMON-PROF-DAG: 1 main {{[0-9a-f]+}} 1 funcJ 0 0 1
CHECK-COMMON-PROF-DAG: 1 main {{[0-9a-f]+}} 1 funcK 0 0 1
CHECK-COMMON-PROF-DAG: 1 main {{[0-9a-f]+}} 1 funcL 0 0 1
CHECK-COMMON-PROF-DAG: 1 main {{[0-9a-f]+}} 1 funcM 0 0 1
CHECK-COMMON-PROF-DAG: 1 main {{[0-9a-f]+}} 1 funcN 0 0 1
CHECK-COMMON-PROF-DAG: 1 main {{[0-9a-f]+}} 1 funcO 0 0 1
CHECK-COMMON-PROF-DAG: 1 main {{[0-9a-f]+}} 1 funcP 0 0 1

RUN: llvm-bolt %t.exe --instrument --instrumentation-file=%t \
RUN:   --instrumentation-file-append-pid \
RUN:   -o %t.instrumented

RUN: %t.instrumented > %t.output
# Wait till output is fully written in case child outlives parent
RUN: bash %S/wait_file.sh %t.output
# Make sure all functions were called
RUN: cat %t.output | FileCheck %s --check-prefix=CHECK-OUTPUT

RUN: %python %S/copy_file.py %t funcA child
RUN: %python %S/copy_file.py %t funcB parent

# Instrumented binary must produce two profiles with only local calls
# recorded. Functions called only in child should not appear in parent's
# process and vice versa.
RUN: cat %t.child.fdata | FileCheck %s --check-prefix=CHECK-CHILD
RUN: cat %t.child.fdata | FileCheck %s --check-prefix=CHECK-NOCHILD
RUN: cat %t.parent.fdata | FileCheck %s --check-prefix=CHECK-PARENT
RUN: cat %t.parent.fdata | FileCheck %s --check-prefix=CHECK-NOPARENT

CHECK-CHILD-DAG: 1 main {{[0-9a-f]+}} 1 funcA 0 0 1
CHECK-CHILD-DAG: 1 main {{[0-9a-f]+}} 1 funcC 0 0 1
CHECK-CHILD-DAG: 1 main {{[0-9a-f]+}} 1 funcE 0 0 1
CHECK-CHILD-DAG: 1 main {{[0-9a-f]+}} 1 funcG 0 0 1
CHECK-CHILD-DAG: 1 main {{[0-9a-f]+}} 1 funcI 0 0 1
CHECK-CHILD-DAG: 1 main {{[0-9a-f]+}} 1 funcK 0 0 1
CHECK-CHILD-DAG: 1 main {{[0-9a-f]+}} 1 funcM 0 0 1
CHECK-CHILD-DAG: 1 main {{[0-9a-f]+}} 1 funcO 0 0 1

CHECK-NOCHILD-NOT: funcB
CHECK-NOCHILD-NOT: funcD
CHECK-NOCHILD-NOT: funcF
CHECK-NOCHILD-NOT: funcH
CHECK-NOCHILD-NOT: funcJ
CHECK-NOCHILD-NOT: funcL
CHECK-NOCHILD-NOT: funcN
CHECK-NOCHILD-NOT: funcP

CHECK-PARENT-DAG: 1 main {{[0-9a-f]+}} 1 funcB 0 0 1
CHECK-PARENT-DAG: 1 main {{[0-9a-f]+}} 1 funcD 0 0 1
CHECK-PARENT-DAG: 1 main {{[0-9a-f]+}} 1 funcF 0 0 1
CHECK-PARENT-DAG: 1 main {{[0-9a-f]+}} 1 funcH 0 0 1
CHECK-PARENT-DAG: 1 main {{[0-9a-f]+}} 1 funcJ 0 0 1
CHECK-PARENT-DAG: 1 main {{[0-9a-f]+}} 1 funcL 0 0 1
CHECK-PARENT-DAG: 1 main {{[0-9a-f]+}} 1 funcN 0 0 1
CHECK-PARENT-DAG: 1 main {{[0-9a-f]+}} 1 funcP 0 0 1

CHECK-NOPARENT-NOT: funcA
CHECK-NOPARENT-NOT: funcC
CHECK-NOPARENT-NOT: funcE
CHECK-NOPARENT-NOT: funcG
CHECK-NOPARENT-NOT: funcI
CHECK-NOPARENT-NOT: funcK
CHECK-NOPARENT-NOT: funcM
CHECK-NOPARENT-NOT: funcO

 */
