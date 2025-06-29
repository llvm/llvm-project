// Check that a driver command line with multiple translation units can create two 
// frontend invocations which can successfully handshake with the module build daemon

// RUN: %kill-process "-cc1modbuildd mbd-multiple-tu"
// RUN: rm -rf mbd-multiple-tu %t
// RUN: split-file %s %t

//--- foo.c
int foo() {return 0;}

//--- main.c
int foo();
int main() {return foo();}

// Return true regardless so lit test does not exit before daemon is killed
// RUN: %clang -fmodule-build-daemon=mbd-multiple-tu -Rmodule-build-daemon %t/main.c %t/foo.c &> %t/output || true
// RUN: %kill-process "-cc1modbuildd mbd-multiple-tu"

// RUN: cat %t/output |  sed 's:\\\\\?:/:g' | FileCheck %s

// CHECK: remark: successfully spawned module build daemon [-Rmodule-build-daemon]
// CHECK-NEXT: remark: successfully connected to module build daemon at mbd-multiple-tu/mbd.sock [-Rmodule-build-daemon]
// CHECK-DAG: remark: clang invocation responsible for {{.*main.c}} successfully completed handshake with module build daemon [-Rmodule-build-daemon]
// CHECK-DAG: remark: clang invocation responsible for {{.*foo.c}} successfully completed handshake with module build daemon [-Rmodule-build-daemon]

// Make sure mbd.err is empty
// RUN: [ ! -s "mbd-launch/mbd.err" ] && true || false
