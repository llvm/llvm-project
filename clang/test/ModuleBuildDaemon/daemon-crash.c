// Check that the module build daemon can create a unix socket after a crash

// RUN: rm -rf mbd-crash %t

// timeout should exit with status 124 which is treated as a failure by lit on 
// windows. Ideally we would like to check the exit code and only return true
// if it equals 124 but lit does not support global bash sysmbols like $?

// RUN: timeout --signal=SIGKILL 2 %clang -cc1modbuildd mbd-crash -v || true
// RUN: timeout --signal=SIGTERM 2 %clang -cc1modbuildd mbd-crash -v || true
// RUN: cat mbd-crash/mbd.out | sed 's:\\\\\?:/:g' | FileCheck %s

// There should only be one shutdown log line due to the crash
// CHECK: MBD created and bound to socket at: mbd-crash/mbd.sock
// CHECK-NEXT: Removing ineligible file: mbd-crash/mbd.sock
// CHECK-NEXT: MBD created and bound to socket at: mbd-crash/mbd.sock
// CHECK-NEXT: Signal received, shutting down

// Make sure mbd.err is empty
// RUN: [ ! -s "mbd-launch/mbd.err" ] && true || false
