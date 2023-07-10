// REQUIRES: !system-windows

// RUN: if pgrep -f "cc1modbuildd mbd-launch"; then pkill -f "cc1modbuildd mbd-launch"; fi

// RUN: %clang -cc1modbuildd mbd-launch -v
// RUN: cat mbd-launch/mbd.out | FileCheck %s -DPREFIX=%t

// CHECK: mbd created and binded to socket address at: mbd-launch/mbd.sock

// RUN: if pgrep -f "cc1modbuildd mbd-launch"; then pkill -f "cc1modbuildd mbd-launch"; fi