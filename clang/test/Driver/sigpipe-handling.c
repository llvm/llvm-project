// REQUIRES: shell
// RUN: %clang -E -fno-integrated-cc1 %s | head | FileCheck %s

// Test that the parent clang driver process doesn't crash when the child cc1
// process receives a SIGPIPE (Unix-only).
//
// The child should exit with IO_ERR, and the parent should exit cleanly.

// CHECK: sigpipe-handling.c
