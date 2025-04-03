// Simple integration test for contextual instrumentation
//
// Copy the header defining ContextNode.
// RUN: mkdir -p %t_include
// RUN: cp %llvm_src/include/llvm/ProfileData/CtxInstrContextNode.h %t_include/
//
// Compile with ctx instrumentation "on". We treat "theRoot" as callgraph root.
// RUN: %clangxx %s %ctxprofilelib -I%t_include -O2 -o %t.bin -mllvm -profile-context-root=theRoot \
// RUN:   -mllvm -ctx-prof-skip-callsite-instr=skip_me
//
// Run the binary, and observe the profile fetch handler's output.
// RUN: %t.bin | FileCheck %s

#include "CtxInstrContextNode.h"
#include <cstdio>
#include <iostream>

using namespace llvm::ctx_profile;
extern "C" void __llvm_ctx_profile_start_collection();
extern "C" bool __llvm_ctx_profile_fetch(ProfileWriter &);

// avoid name mangling
extern "C" {
__attribute__((noinline)) void skip_me() {}

__attribute__((noinline)) void someFunction(int I) {
  if (I % 2)
    printf("check odd\n");
  else
    printf("check even\n");
  skip_me();
}

// block inlining because the pre-inliner otherwise will inline this - it's
// too small.
__attribute__((noinline)) void theRoot() {
  printf("check 1\n");
  someFunction(1);
#pragma nounroll
  for (auto I = 0; I < 2; ++I) {
    someFunction(I);
  }
  skip_me();
}

__attribute__((noinline)) void flatFct() {
  printf("flat check 1\n");
  someFunction(1);
#pragma nounroll
  for (auto I = 0; I < 2; ++I) {
    someFunction(I);
  }
}
}

// Make sure the program actually ran correctly.
// CHECK: check 1
// CHECK-NEXT: check odd
// CHECK-NEXT: check even
// CHECK-NEXT: check odd
// CHECK-NEXT: flat check 1
// CHECK-NEXT: check odd
// CHECK-NEXT: check even
// CHECK-NEXT: check odd

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
    printProfile(RootNode, "", "");
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

// 8657661246551306189 is theRoot. We expect 2 callsites and 2 counters - one
// for the entry basic block and one for the loop.
// 6759619411192316602 is someFunction. We expect all context instances to show
// the same nr of counters and callsites, but the counters will be different.
// The first context is for the first callsite with theRoot as parent, and the
// second counter in someFunction will be 0 (we pass an odd nr, and the other
// path gets instrumented).
// The second context is in the loop. We expect 2 entries and each of the
// branches would be taken once, so the second counter is 1.
// CHECK-NEXT: Entered Context Section
// CHECK-NEXT: Entering Root 8657661246551306189 with total entry count 1
// skip_me is entered 4 times: 3 via `someFunction`, and once from `theRoot`
// CHECK-NEXT: Unhandled GUID: 17928815489886282963 entered 4 times
// CHECK-NEXT: Guid: 8657661246551306189
// CHECK-NEXT: Entries: 1
// CHECK-NEXT: 2 counters and 3 callsites
// CHECK-NEXT: Counter values: 1 2
// CHECK-NEXT: At Index 1:
// CHECK-NEXT:   Guid: 6759619411192316602
// CHECK-NEXT:   Entries: 1
// CHECK-NEXT:   2 counters and 2 callsites
// CHECK-NEXT:   Counter values: 1 0
// CHECK-NEXT: At Index 2:
// CHECK-NEXT:   Guid: 6759619411192316602
// CHECK-NEXT:   Entries: 2
// CHECK-NEXT:   2 counters and 2 callsites
// CHECK-NEXT:   Counter values: 2 1
// CHECK-NEXT: Exited Context Section
// CHECK-NEXT: Entered Flat Section
// This is `skip_me`. Entered 3 times via `someFunction`
// CHECK-NEXT: Flat: 17928815489886282963 3
// CHECK-NEXT: Flat: 6759619411192316602 3,1
// This is flatFct (guid: 14569438697463215220)
// CHECK-NEXT: Flat: 14569438697463215220 1,2
// CHECK-NEXT: Exited Flat Section

bool profileWriter() {
  TestProfileWriter W;
  return __llvm_ctx_profile_fetch(W);
}

int main(int argc, char **argv) {
  __llvm_ctx_profile_start_collection();
  theRoot();
  flatFct();
  // This would be implemented in a specific RPC handler, but here we just call
  // it directly.
  return !profileWriter();
}
