// Test that asan_symbolize does not hang when provided with an non-existing
// path.
// RUN: echo '#0 0xabcdabcd (%t/bad/path+0x1234)' | %asan_symbolize | FileCheck %s
// CHECK: #0 0xabcdabcd
