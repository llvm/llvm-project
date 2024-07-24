// Check that the module build daemon can create a unix socket

// RUN: rm -rf mbd-launch %t

// timeout should exit with status 124 which is treated as a failure by lit on 
// windows. Ideally we would like to check the exit code and only return true
// if it equals 124 but lit does not support global bash symbols like $?

// RUN: timeout --signal=SIGTERM 2 %clang -cc1modbuildd mbd-launch -v || true
// RUN: cat mbd-launch/mbd.out | sed 's:\\\\\?:/:g' | FileCheck %s

// CHECK: MBD created and bound to socket at: mbd-launch/mbd.sock
// CHECK-NEXT: Signal received, shutting down

// Make sure mbd.err is empty
// RUN: [ ! -s "mbd-launch/mbd.err" ] && true || false
