//===-- mutex_test.cc--------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mutex.h"

#include "gtest/gtest.h"

#include <string.h>

template <typename MutexType> class TestData {
public:
  explicit TestData(MutexType *M) : Mutex(M) {
    for (scudo::u32 I = 0; I < Size; I++)
      Data[I] = 0;
  }

  void write() {
    Lock L(Mutex);
    T V0 = Data[0];
    for (scudo::u32 I = 0; I < Size; I++) {
      EXPECT_EQ(Data[I], V0);
      Data[I]++;
    }
  }

  void tryWrite() {
    if (!Mutex->tryLock())
      return;
    T V0 = Data[0];
    for (scudo::u32 I = 0; I < Size; I++) {
      EXPECT_EQ(Data[I], V0);
      Data[I]++;
    }
    Mutex->unlock();
  }

  void backoff() {
    volatile T LocalData[Size] = {};
    for (scudo::u32 I = 0; I < Size; I++) {
      LocalData[I]++;
      EXPECT_EQ(LocalData[I], 1U);
    }
  }

private:
  typedef scudo::GenericScopedLock<MutexType> Lock;
  static const scudo::u32 Size = 64U;
  typedef scudo::u64 T;
  MutexType *Mutex;
  ALIGNED(SCUDO_CACHE_LINE_SIZE) T Data[Size];
};

const scudo::u32 NumberOfThreads = 8;
#if SCUDO_DEBUG
const scudo::u32 NumberOfIterations = 4 * 1024;
#else
const scudo::u32 NumberOfIterations = 16 * 1024;
#endif

template <typename MutexType> static void *lockThread(void *Param) {
  TestData<MutexType> *Data = reinterpret_cast<TestData<MutexType> *>(Param);
  for (scudo::u32 I = 0; I < NumberOfIterations; I++) {
    Data->write();
    Data->backoff();
  }
  return 0;
}

template <typename MutexType> static void *tryThread(void *Param) {
  TestData<MutexType> *Data = reinterpret_cast<TestData<MutexType> *>(Param);
  for (scudo::u32 I = 0; I < NumberOfIterations; I++) {
    Data->tryWrite();
    Data->backoff();
  }
  return 0;
}

template <typename MutexType> static void checkLocked(MutexType *M) {
  scudo::GenericScopedLock<MutexType> L(M);
  M->checkLocked();
}

TEST(ScudoMutexTest, SpinMutex) {
  scudo::SpinMutex M;
  M.init();
  TestData<scudo::SpinMutex> Data(&M);
  pthread_t Threads[NumberOfThreads];
  for (scudo::u32 I = 0; I < NumberOfThreads; I++)
    pthread_create(&Threads[I], 0, lockThread<scudo::SpinMutex>, &Data);
  for (scudo::u32 I = 0; I < NumberOfThreads; I++)
    pthread_join(Threads[I], 0);
}

TEST(ScudoMutexTest, SpinMutexTry) {
  scudo::SpinMutex M;
  M.init();
  TestData<scudo::SpinMutex> Data(&M);
  pthread_t Threads[NumberOfThreads];
  for (scudo::u32 I = 0; I < NumberOfThreads; I++)
    pthread_create(&Threads[I], 0, tryThread<scudo::SpinMutex>, &Data);
  for (scudo::u32 I = 0; I < NumberOfThreads; I++)
    pthread_join(Threads[I], 0);
}

TEST(ScudoMutexTest, BlockingMutex) {
  scudo::u64 MutexMemory[1024] = {};
  scudo::BlockingMutex *M =
      new (MutexMemory) scudo::BlockingMutex(scudo::LINKER_INITIALIZED);
  TestData<scudo::BlockingMutex> Data(M);
  pthread_t Threads[NumberOfThreads];
  for (scudo::u32 I = 0; I < NumberOfThreads; I++)
    pthread_create(&Threads[I], 0, lockThread<scudo::BlockingMutex>, &Data);
  for (scudo::u32 I = 0; I < NumberOfThreads; I++)
    pthread_join(Threads[I], 0);
  checkLocked(M);
}
