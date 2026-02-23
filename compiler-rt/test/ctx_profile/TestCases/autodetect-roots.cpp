// Root autodetection test for contextual profiling
//
// Copy the header defining ContextNode.
// RUN: mkdir -p %t_include
// RUN: cp %llvm_src/include/llvm/ProfileData/CtxInstrContextNode.h %t_include/
//
// Compile with ctx instrumentation "on". We use -profile-context-root as signal
// that we want contextual profiling, but we can specify anything there, that
// won't be matched with any function, and result in the behavior we are aiming
// for here.
//
// RUN: %clangxx %s %ctxprofilelib -I%t_include -O2 -o %t.bin \
// RUN:   -mllvm -profile-context-root="<autodetect>" -g -Wl,-export-dynamic
//
// Run the binary, and observe the profile fetch handler's output.
// RUN: %t.bin > %t.log
// The check is split because the root order is non-deterministic.
// RUN: cat %t.log | FileCheck %s
// RUN: cat %t.log | FileCheck %s --check-prefix=CHECK-ROOT1
// RUN: cat %t.log | FileCheck %s --check-prefix=CHECK-ROOT2

#include "CtxInstrContextNode.h"
#include <atomic>
#include <cstdio>
#include <iostream>
#include <thread>

using namespace llvm::ctx_profile;
extern "C" void __llvm_ctx_profile_start_collection(unsigned);
extern "C" bool __llvm_ctx_profile_fetch(ProfileWriter &);

// avoid name mangling
extern "C" {
__attribute__((noinline)) void anotherFunction() {}
__attribute__((noinline)) void mock1() {}
__attribute__((noinline)) void mock2() {}
__attribute__((noinline)) void someFunction(int I) {
  if (I % 2)
    mock1();
  else
    mock2();
  anotherFunction();
}

// block inlining because the pre-inliner otherwise will inline this - it's
// too small.
__attribute__((noinline)) void theRoot() {
  someFunction(1);
#pragma nounroll
  for (auto I = 0; I < 2; ++I) {
    someFunction(I);
  }
  anotherFunction();
}
}

class TestProfileWriter : public ProfileWriter {
  void printProfile(const ContextNode &Node, const std::string &Indent,
                    const std::string &Increment) {
    std::cout << Indent << "Guid: " << Node.guid() << std::endl;
    std::cout << Indent << "Entries: " << Node.entrycount() << std::endl;
    std::cout << Indent << Node.counters_size() << " counters and "
              << Node.callsites_size() << " callsites" << std::endl;
    std::cout << Indent << "Counter values: ";
    for (uint32_t I = 0U; I < Node.counters_size(); ++I)
      std::cout << Node.counters()[I] << " ";
    std::cout << std::endl;
    for (uint32_t I = 0U; I < Node.callsites_size(); ++I)
      for (const auto *N = Node.subContexts()[I]; N; N = N->next()) {
        std::cout << Indent << "At Index " << I << ":" << std::endl;
        printProfile(*N, Indent + Increment, Increment);
      }
  }

  void startContextSection() override {
    std::cout << "Entered Context Section" << std::endl;
  }

  void endContextSection() override {
    std::cout << "Exited Context Section" << std::endl;
  }

  void writeContextual(const ContextNode &RootNode,
                       const ContextNode *Unhandled,
                       uint64_t EntryCount) override {
    std::cout << "Entering Root " << RootNode.guid()
              << " with total entry count " << EntryCount << std::endl;
    for (const auto *P = Unhandled; P; P = P->next())
      std::cout << "Unhandled GUID: " << P->guid() << " entered "
                << P->entrycount() << " times" << std::endl;
    printProfile(RootNode, " ", " ");
  }

  void startFlatSection() override {
    std::cout << "Entered Flat Section" << std::endl;
  }

  void writeFlat(GUID Guid, const uint64_t *Buffer,
                 size_t BufferSize) override {
    std::cout << "Flat: " << Guid << " " << Buffer[0];
    for (size_t I = 1U; I < BufferSize; ++I)
      std::cout << "," << Buffer[I];
    std::cout << std::endl;
  };

  void endFlatSection() override {
    std::cout << "Exited Flat Section" << std::endl;
  }
};

// Guid:3950394326069683896 is anotherFunction
// Guid:6759619411192316602 is someFunction
// These are expected to be the auto-detected roots. This is because we cannot
// discern (with the current autodetection mechanism) if theRoot
// (Guid:8657661246551306189) is ever re-entered.
//
// CHECK: Entered Context Section
// CHECK: Entering Root
// CHECK: Entering Root
// CHECK-NOT: Entering Root
// CHECK-ROOT1: Entering Root 6759619411192316602 with total entry count {{[0-9]+}}
// CHECK-ROOT1-NEXT: Guid: 6759619411192316602
// CHECK-ROOT1-NEXT:  Entries: [[ROOT1_COUNTER1:[0-9]+]]
// CHECK-ROOT1-NEXT:  2 counters and 3 callsites
// CHECK-ROOT1-NEXT:  Counter values: [[ROOT1_COUNTER1]] [[ROOT1_COUNTER2:[0-9]+]]
// CHECK-ROOT1-NEXT:  At Index 0:
// CHECK-ROOT1-NEXT:   Guid: 434762725428799310
// CHECK-ROOT1-NEXT:   Entries: [[ROOT1_COUNTER3:[0-9]+]]
// CHECK-ROOT1-NEXT:   1 counters and 0 callsites
// CHECK-ROOT1-NEXT:   Counter values: [[ROOT1_COUNTER3]]
// CHECK-ROOT1-NEXT:  At Index 1:
// CHECK-ROOT1-NEXT:   Guid: 5578595117440393467
// CHECK-ROOT1-NEXT:   Entries: [[ROOT1_COUNTER2]]
// CHECK-ROOT1-NEXT:   1 counters and 0 callsites
// CHECK-ROOT1-NEXT:   Counter values: [[ROOT1_COUNTER2]]
// CHECK-ROOT1-NEXT:  At Index 2:
// CHECK-ROOT1-NEXT:   Guid: 3950394326069683896
// CHECK-ROOT1-NEXT:   Entries: [[ROOT1_COUNTER1]]
// CHECK-ROOT1-NEXT:   1 counters and 0 callsites
// CHECK-ROOT1-NEXT:   Counter values: [[ROOT1_COUNTER1]]
// CHECK-ROOT2: Entering Root 3950394326069683896 with total entry count {{[0-9]+}}
// CHECK-ROOT2-NEXT:  Guid: 3950394326069683896
// CHECK-ROOT2-NEXT:  Entries: [[ROOT2_COUNTER:[0-9]+]]
// CHECK-ROOT2-NEXT:  1 counters and 0 callsites
// CHECK-ROOT2-NEXT:  Counter values: [[ROOT2_COUNTER]]
// CHECK: Exited Context Section
// CHECK-NEXT: Entered Flat Section
// CHECK-DAG: Flat: 434762725428799310 {{[0-9]+}}
// CHECK-DAG: Flat: 5578595117440393467 {{[0-9]+}}
// CHECK-DAG: Flat: 8657661246551306189 {{[0-9]+}},{{[0-9]+}}
// CHECK-DAG: Flat: {{[0-9]+}} 1
// CHECK-DAG: Flat: {{[0-9]+}} 1
// CHECK-DAG: Flat: {{[0-9]+}} 1
// CHECK-NEXT: Exited Flat Section

bool profileWriter() {
  TestProfileWriter W;
  return __llvm_ctx_profile_fetch(W);
}

int main(int argc, char **argv) {
  std::atomic<bool> Stop = false;
  std::atomic<int> Started = 0;
  std::thread T1([&]() {
    ++Started;
    while (!Stop) {
      theRoot();
    }
  });

  std::thread T2([&]() {
    ++Started;
    while (!Stop) {
      theRoot();
    }
  });

  std::thread T3([&]() {
    while (Started < 2) {
    }
    __llvm_ctx_profile_start_collection(5);
  });

  T3.join();
  using namespace std::chrono_literals;

  std::this_thread::sleep_for(10s);
  Stop = true;
  T1.join();
  T2.join();

  // This would be implemented in a specific RPC handler, but here we just call
  // it directly.
  return !profileWriter();
}
