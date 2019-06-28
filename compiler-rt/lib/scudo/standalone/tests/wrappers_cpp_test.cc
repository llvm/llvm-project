//===-- wrappers_cpp_test.cc ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <condition_variable>
#include <mutex>
#include <thread>

// Note that every Cxx allocation function in the test binary will be fulfilled
// by Scudo. See the comment in the C counterpart of this file.

extern "C" __attribute__((visibility("default"))) const char *
__scudo_default_options() {
  return "quarantine_size_kb=256:thread_local_quarantine_size_kb=128:"
         "quarantine_max_chunk_size=512:dealloc_type_mismatch=true";
}

template <typename T> static void testCxxNew() {
  T *P = new T;
  EXPECT_NE(P, nullptr);
  memset(P, 0x42, sizeof(T));
  EXPECT_DEATH(delete[] P, "");
  delete P;
  EXPECT_DEATH(delete P, "");

  P = new T;
  EXPECT_NE(P, nullptr);
  memset(P, 0x42, sizeof(T));
  operator delete(P, sizeof(T));

  P = new (std::nothrow) T;
  EXPECT_NE(P, nullptr);
  memset(P, 0x42, sizeof(T));
  delete P;

  const size_t N = 16U;
  T *A = new T[N];
  EXPECT_NE(A, nullptr);
  memset(A, 0x42, sizeof(T) * N);
  EXPECT_DEATH(delete A, "");
  delete[] A;
  EXPECT_DEATH(delete[] A, "");

  A = new T[N];
  EXPECT_NE(A, nullptr);
  memset(A, 0x42, sizeof(T) * N);
  operator delete[](A, sizeof(T) * N);

  A = new (std::nothrow) T[N];
  EXPECT_NE(A, nullptr);
  memset(A, 0x42, sizeof(T) * N);
  delete[] A;
}

class Pixel {
public:
  enum class Color { Red, Green, Blue };
  int X = 0;
  int Y = 0;
  Color C = Color::Red;
};

TEST(ScudoWrappersCppTest, New) {
  testCxxNew<bool>();
  testCxxNew<uint8_t>();
  testCxxNew<uint16_t>();
  testCxxNew<uint32_t>();
  testCxxNew<uint64_t>();
  testCxxNew<float>();
  testCxxNew<double>();
  testCxxNew<long double>();
  testCxxNew<Pixel>();
}

static std::mutex Mutex;
static std::condition_variable Cv;
static bool Ready = false;

static void stressNew() {
  std::vector<uintptr_t *> V;
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    while (!Ready)
      Cv.wait(Lock);
  }
  for (size_t I = 0; I < 256U; I++) {
    const size_t N = std::rand() % 128U;
    uintptr_t *P = new uintptr_t[N];
    if (P) {
      memset(P, 0x42, sizeof(uintptr_t) * N);
      V.push_back(P);
    }
  }
  while (!V.empty()) {
    delete[] V.back();
    V.pop_back();
  }
}

TEST(ScudoWrappersCppTest, ThreadedNew) {
  std::thread Threads[32];
  for (size_t I = 0U; I < sizeof(Threads) / sizeof(Threads[0]); I++)
    Threads[I] = std::thread(stressNew);
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    Ready = true;
    Cv.notify_all();
  }
  for (auto &T : Threads)
    T.join();
}
