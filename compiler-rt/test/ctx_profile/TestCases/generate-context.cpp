// Simple integration test for contextual instrumentation
//
// Copy the header defining ContextNode.
// RUN: mkdir -p %t_include
// RUN: cp %llvm_src/include/llvm/ProfileData/CtxInstrContextNode.h %t_include/
//
// Compile with ctx instrumentation "on". We treat "the_root" as callgraph root.
// RUN: %clangxx %s -lclang_rt.ctx_profile -I%t_include -O2 -o %t.bin -mllvm -profile-context-root=the_root
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

extern "C" {
__attribute__((noinline)) void someFunction() { printf("check 2\n"); }

// block inlining because the pre-inliner otherwise will inline this - it's
// too small.
__attribute__((noinline)) void the_root() {
  printf("check 1\n");
  someFunction();
  someFunction();
}
}

// Make sure the program actually ran correctly.
// CHECK: check 1
// CHECK: check 2
// CHECK: check 2

void printProfile(const ContextNode &Node, const std::string &Indent,
                  const std::string &Increment) {
  std::cout << Indent << "Guid: " << Node.guid() << std::endl;
  std::cout << Indent << "Entries: " << Node.entrycount() << std::endl;
  for (uint32_t I = 0U; I < Node.callsites_size(); ++I)
    for (const auto *N = Node.subContexts()[I]; N; N = N->next()) {
      std::cout << Indent << "At Index " << I << ":" << std::endl;
      printProfile(*N, Indent + Increment, Increment);
    }
}

// CHECK: Guid: 11065787667334760794
// CHECK: Entries: 1
// CHECK: At Index 1:
// CHECK:   Guid: 6759619411192316602
// CHECK:   Entries: 1
// CHECK: At Index 2:
// CHECK:   Guid: 6759619411192316602
// CHECK:   Entries: 1

bool profileWriter() {
  return __llvm_ctx_profile_fetch(
      nullptr, +[](void *, const ContextNode &Node) {
        printProfile(Node, "", "  ");
        return true;
      });
}

int main(int argc, char **argv) {
  the_root();
  // This would be implemented in a specific RPC handler, but here we just call
  // it directly.
  return !profileWriter();
}
