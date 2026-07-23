// Test that the clang driver exits via signal when cc1 runs out-of-process
// (-fno-integrated-cc1) and crashes.
//
// RUN: not %crash_opt %clang %s -fsyntax-only -fno-integrated-cc1 2>&1
//
// REQUIRES: crash-recovery

#pragma clang __debug parser_crash
