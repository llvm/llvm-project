// Check hard_rss_limit_mb. Not all sanitizers implement it yet.
// RUN: %clangxx -O2 %s -o %t
//
// Run with limit should fail:
// RUN: %env_tool_opts=hard_rss_limit_mb=100                           not %run %t 2>&1 | FileCheck %s
// This run uses getrusage:
// RUN: %env_tool_opts=hard_rss_limit_mb=100:can_use_proc_maps_statm=0 not %run %t 2>&1 | FileCheck %s
//
// Run w/o limit or with a large enough limit should pass:
// RUN: %env_tool_opts=hard_rss_limit_mb=1000 %run %t
// RUN: %run %t
//
// FIXME: make it work for other sanitizers.
// XFAIL: lsan
// XFAIL: tsan
// XFAIL: msan
// XFAIL: ubsan

// https://github.com/google/sanitizers/issues/981
// UNSUPPORTED: android-26

#include <string.h>
#include <stdio.h>
#include <unistd.h>

const int kNumAllocs = 200 * 1000;
const int kAllocSize = 1000;
volatile char *sink[kNumAllocs];

int main(int argc, char **argv) {
  for (int i = 0; i < kNumAllocs; i++) {
    if ((i % 1000) == 0) {
      // Don't write to stderr! Doing that triggers a kernel race condition
      // between this thread and the rss-limit thread, and may lose part of the
      // output. See https://lkml.org/lkml/2014/2/17/324.
      printf("[%d]\n", i);
    }
    char *x = new char[kAllocSize];
    memset(x, 0, kAllocSize);
    sink[i] = x;
  }
  sleep(1);  // Make sure the background thread has time to kill the process.
// CHECK: hard rss limit exhausted
}
