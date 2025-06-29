// Check that a clang invocation can spawn and handshake with a module build daemon

// RUN: %kill-process "-cc1modbuildd mbd-handshake"
// RUN: rm -rf mbd-handshake %t
// RUN: split-file %s %t

//--- main.c
int main() {return 0;}

// Return true regardless so lit test does not exit before daemon is killed
// RUN: %clang -fmodule-build-daemon=mbd-handshake -Rmodule-build-daemon %t/main.c &> %t/output-new || true
// RUN: %clang -fmodule-build-daemon=mbd-handshake -Rmodule-build-daemon %t/main.c &> %t/output-existing || true
// RUN: %kill-process "-cc1modbuildd mbd-handshake"

// RUN: cat %t/output-new |  sed 's:\\\\\?:/:g' | FileCheck %s
// RUN: cat %t/output-existing |  sed 's:\\\\\?:/:g' | FileCheck %s --check-prefix=CHECK-EXIST

// CHECK: remark: successfully spawned module build daemon [-Rmodule-build-daemon]
// CHECK-NEXT: remark: successfully connected to module build daemon at mbd-handshake/mbd.sock [-Rmodule-build-daemon]
// CHECK-NEXT: remark: clang invocation responsible for {{.*main.c}} successfully completed handshake with module build daemon [-Rmodule-build-daemon]

// Check that a clang invocation can handshake with an existing module build daemon
// CHECK-EXIST: remark: clang invocation responsible for {{.*main.c}} successfully completed handshake with module build daemon [-Rmodule-build-daemon]

// Make sure mbd.err is empty
// RUN: [ ! -s "mbd-launch/mbd.err" ] && true || false
