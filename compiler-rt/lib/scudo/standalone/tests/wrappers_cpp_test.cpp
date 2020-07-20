//===-- wrappers_cpp_test.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tests/scudo_unit_test.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

void operator delete(void *, size_t) noexcept;
void operator delete[](void *, size_t) noexcept;

// Note that every Cxx allocation function in the test binary will be fulfilled
// by Scudo. See the comment in the C counterpart of this file.

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

#if !SCUDO_FUCHSIA
// TODO(kostyak): for me, this test fails in a specific configuration when ran
//                by itself with some Scudo or GWP-ASan violation. Other people
//                can't seem to reproduce the failure. Consider skipping this in
//                the event it fails on the upstream bots.
TEST(ScudoWrappersCppTest, AllocAfterFork) {
  std::atomic_bool Stop;

  // Create threads that simply allocate and free different sizes.
  std::vector<std::thread *> Threads;
  for (size_t N = 0; N < 5; N++) {
    std::thread *T = new std::thread([&Stop] {
      while (!Stop) {
        for (size_t SizeLog = 3; SizeLog <= 21; SizeLog++) {
          char *P = new char[1UL << SizeLog];
          EXPECT_NE(P, nullptr);
          // Make sure this value is not optimized away.
          asm volatile("" : : "r,m"(P) : "memory");
          delete[] P;
        }
      }
    });
    Threads.push_back(T);
  }

  // Create a thread to fork and allocate.
  for (size_t N = 0; N < 100; N++) {
    pid_t Pid;
    if ((Pid = fork()) == 0) {
      for (size_t SizeLog = 3; SizeLog <= 21; SizeLog++) {
        char *P = new char[1UL << SizeLog];
        EXPECT_NE(P, nullptr);
        // Make sure this value is not optimized away.
        asm volatile("" : : "r,m"(P) : "memory");
        // Make sure we can touch all of the allocation.
        memset(P, 0x32, 1U << SizeLog);
        // EXPECT_LE(1U << SizeLog, malloc_usable_size(ptr));
        delete[] P;
      }
      _exit(10);
    }
    EXPECT_NE(-1, Pid);
    int Status;
    EXPECT_EQ(Pid, waitpid(Pid, &Status, 0));
    EXPECT_FALSE(WIFSIGNALED(Status));
    EXPECT_EQ(10, WEXITSTATUS(Status));
  }

  printf("Waiting for threads to complete\n");
  Stop = true;
  for (auto Thread : Threads)
    Thread->join();
  Threads.clear();
}
#endif
