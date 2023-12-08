// RUN: %clang_cc1 nul

// REQUIRES: system-windows

// Verify that cc1 doesn't crash with an assertion failure
// in MemoryBuffer.cpp due to an invalid file size reported
// when the Windows 'nul' device is passed in input.

