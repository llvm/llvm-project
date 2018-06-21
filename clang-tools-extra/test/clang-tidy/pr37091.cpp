// REQUIRES: shell
// RUN: rm -rf %t
// RUN: mkdir -p %t

// This is a reproducer for PR37091.
//
// Verify that no temporary files are left behind by the clang-tidy invocation.

// RUN: env TMPDIR=%t TEMP=%t TMP=%t clang-tidy %s -- --target=mips64
// RUN: rmdir %t
