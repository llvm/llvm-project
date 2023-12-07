// RUN: %clang_dfsan -DTRACK_ORIGINS=2 -mllvm -dfsan-track-origins=2 %s -o %t && %run %t
// RUN: %clang_dfsan -DTRACK_ORIGINS=1 -mllvm -dfsan-track-origins=1 %s -o %t && %run %t
// RUN: %clang_dfsan -DTRACK_ORIGINS=0 %s -o %t && %run %t

#include <sanitizer/dfsan_interface.h>

#include <assert.h>

int main(int argc, char *argv[]) {
  assert(dfsan_get_track_origins() == TRACK_ORIGINS);
}
