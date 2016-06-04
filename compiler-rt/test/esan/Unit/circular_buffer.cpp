// RUN: %clangxx_unit -O0 %s -o %t 2>&1
// RUN: %env_esan_opts="record_snapshots=0" %run %t 2>&1 | FileCheck %s

#include "esan/esan_circular_buffer.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include <assert.h>
#include <stdio.h>

static const int TestBufCapacity = 4;

// The buffer should have a capacity of TestBufCapacity.
void testBuffer(__esan::CircularBuffer<int> *Buf) {
  assert(Buf->size() == 0);
  assert(Buf->empty());

  Buf->push_back(1);
  assert(Buf->back() == 1);
  assert((*Buf)[0] == 1);
  assert(Buf->size() == 1);
  assert(!Buf->empty());

  Buf->push_back(2);
  Buf->push_back(3);
  Buf->push_back(4);
  Buf->push_back(5);
  assert((*Buf)[0] == 2);
  assert(Buf->size() == 4);

  Buf->pop_back();
  assert((*Buf)[0] == 2);
  assert(Buf->size() == 3);

  Buf->pop_back();
  Buf->pop_back();
  assert((*Buf)[0] == 2);
  assert(Buf->size() == 1);
  assert(!Buf->empty());

  Buf->pop_back();
  assert(Buf->empty());
}

int main()
{
  // Test initialize/free.
  __esan::CircularBuffer<int> GlobalBuf;
  GlobalBuf.initialize(TestBufCapacity);
  testBuffer(&GlobalBuf);
  GlobalBuf.free();

  // Test constructor/free.
  __esan::CircularBuffer<int> *LocalBuf;
  static char placeholder[sizeof(*LocalBuf)];
  LocalBuf = new(placeholder) __esan::CircularBuffer<int>(TestBufCapacity);
  testBuffer(LocalBuf);
  LocalBuf->free();

  fprintf(stderr, "All checks passed.\n");
  // CHECK: All checks passed.
  return 0;
}
