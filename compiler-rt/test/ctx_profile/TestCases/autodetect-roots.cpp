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
// RUN %t.bin | FileCheck %s

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
// CHECK:      Entered Context Section
// CHECK-NEXT: Entering Root 6759619411192316602 with total entry count 12463157
// CHECK-NEXT: Guid: 6759619411192316602
// CHECK-NEXT:  Entries: 5391142
// CHECK-NEXT:  2 counters and 3 callsites
// CHECK-NEXT:  Counter values: 5391142 1832357
// CHECK-NEXT:  At Index 0:
// CHECK-NEXT:   Guid: 434762725428799310
// CHECK-NEXT:   Entries: 3558785
// CHECK-NEXT:   1 counters and 0 callsites
// CHECK-NEXT:   Counter values: 3558785
// CHECK-NEXT:  At Index 1:
// CHECK-NEXT:   Guid: 5578595117440393467
// CHECK-NEXT:   Entries: 1832357
// CHECK-NEXT:   1 counters and 0 callsites
// CHECK-NEXT:   Counter values: 1832357
// CHECK-NEXT:  At Index 2:
// CHECK-NEXT:   Guid: 3950394326069683896
// CHECK-NEXT:   Entries: 5391142
// CHECK-NEXT:   1 counters and 0 callsites
// CHECK-NEXT:   Counter values: 5391142
// CHECK-NEXT: Entering Root 3950394326069683896 with total entry count 11226401
// CHECK-NEXT:  Guid: 3950394326069683896
// CHECK-NEXT:  Entries: 10767423
// CHECK-NEXT:  1 counters and 0 callsites
// CHECK-NEXT:  Counter values: 10767423
// CHECK-NEXT: Exited Context Section
// CHECK-NEXT: Entered Flat Section
// CHECK-NEXT: Flat: 2597020043743142491 1
// CHECK-NEXT: Flat: 4321328481998485159 1
// CHECK-NEXT: Flat: 8657661246551306189 9114175,18099613
// CHECK-NEXT: Flat: 434762725428799310 10574815
// CHECK-NEXT: Flat: 5578595117440393467 5265754
// CHECK-NEXT: Flat: 12566320182004153844 1
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
