// Simple integration test for contextual instrumentation
//
// Copy the header defining ContextNode.
// RUN: mkdir -p %t_include
// RUN: cp %llvm_src/include/llvm/ProfileData/CtxInstrContextNode.h %t_include/
//
// Compile with ctx instrumentation "on". We treat "theRoot" as callgraph root.
// RUN: %clangxx %s %ctxprofilelib -I%t_include -O2 -o %t.bin -mllvm -profile-context-root=theRoot
//
// Run the binary, and observe the profile fetch handler's output.
// RUN: %t.bin | FileCheck %s

#include "CtxInstrContextNode.h"
#include <cstdio>
#include <iostream>

using namespace llvm::ctx_profile;
extern "C" bool __llvm_ctx_profile_fetch(void *Data,
                                         bool (*Writer)(void *,
                                                        const ContextNode &));

// avoid name mangling
extern "C" {
__attribute__((noinline)) void someFunction(int I) {
  if (I % 2)
    printf("check odd\n");
  else
    printf("check even\n");
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
}
}

// Make sure the program actually ran correctly.
// CHECK: check 1
// CHECK-NEXT: check odd
// CHECK-NEXT: check even
// CHECK-NEXT: check odd

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

// 8657661246551306189 is theRoot. We expect 2 callsites and 2 counters - one
// for the entry basic block and one for the loop.
// 6759619411192316602 is someFunction. We expect all context instances to show
// the same nr of counters and callsites, but the counters will be different.
// The first context is for the first callsite with theRoot as parent, and the
// second counter in someFunction will be 0 (we pass an odd nr, and the other
// path gets instrumented).
// The second context is in the loop. We expect 2 entries and each of the
// branches would be taken once, so the second counter is 1.
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

bool profileWriter() {
  return __llvm_ctx_profile_fetch(
      nullptr, +[](void *, const ContextNode &Node) {
        printProfile(Node, "", "  ");
        return true;
      });
}

int main(int argc, char **argv) {
  theRoot();
  // This would be implemented in a specific RPC handler, but here we just call
  // it directly.
  return !profileWriter();
}
