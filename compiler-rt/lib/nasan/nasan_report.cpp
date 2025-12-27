//===-- nasan_report.cpp - NoAliasSanitizer Violation Reporting ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Violation reporting for NoAliasSanitizer.
//
//===----------------------------------------------------------------------===//

#include "nasan_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

namespace __nasan {

using namespace __sanitizer;

// Control behavior on error
static bool should_abort_on_error() {
  const char *env = GetEnv("NASAN_OPTIONS");
  if (env) {
    // Simple substring search for abort_on_error=0
    const char *p = env;
    while (*p) {
      if (internal_strncmp(p, "abort_on_error=0", 16) == 0) {
        return false;
      }
      p++;
    }
  }
  return true;  // default: abort
}

static void print_spaces(int count) {
  for (int i = 0; i < count; i++) {
    Printf(" ");
  }
}

static void print_provenance_chain(ProvenanceID prov, int indent) {
  NASanThreadState *state = get_thread_state();

  ProvenanceID current = prov;
  int depth = 0;

  while (current != 0 && depth < 10) {  // Limit depth to prevent loops
    auto *entry = state->provenance_info.find(current);
    if (!entry) {
      print_spaces(indent);
      Printf("[%llu] <unknown provenance>\n", (unsigned long long)current);
      break;
    }

    const ProvenanceInfo &info = entry->second;
    print_spaces(indent);
    Printf("[%llu] ", (unsigned long long)info.id);

    if (info.is_allocation) {
      Printf("allocation at %p\n", info.noalias_param);
    } else {
      Printf("noalias parameter %p in %s\n",
             info.noalias_param, info.function_name);
    }

    print_spaces(indent);
    Printf("  at %s:%d\n", info.location.file, info.location.line);

    if (info.based_on != 0) {
      print_spaces(indent);
      Printf("  based on:\n");
      current = info.based_on;
      indent += 4;
      depth++;
    } else {
      break;
    }
  }
}

static void print_stack_trace() {
  BufferedStackTrace stack;
  stack.Unwind(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME(), nullptr,
               common_flags()->fast_unwind_on_fatal, /* max_depth */ 30);
  stack.Print();
}

void report_violation(
    void *addr,
    const InternalMmapVector<ProvenanceID> &accessing_provs,
    const DenseSet<ProvenanceID> &existing_provs) {

  Printf("\n");
  Printf("==================================\n");
  Printf("ERROR: NoAliasSanitizer: conflicting accesses via incompatible provenances\n");
  Printf("==================================\n");
  Printf("\nMemory address: %p\n", addr);

  Printf("\nAccessing via provenances:\n");
  for (uptr i = 0; i < accessing_provs.size(); i++) {
    ProvenanceID prov = accessing_provs[i];
    if (prov == 0) {
      Printf("  [0] (no provenance)\n");
    } else {
      print_provenance_chain(prov, 2);
    }
  }

  Printf("\nPreviously accessed via provenances:\n");
  existing_provs.forEach([](const ProvenanceID &prov) {
    if (prov == 0) {
      Printf("  [0] (no provenance)\n");
    } else {
      print_provenance_chain(prov, 2);
    }
  });

  Printf("\nStack trace:\n");
  print_stack_trace();

  Printf("\n");
  Printf("SUMMARY: NoAliasSanitizer: noalias-violation\n");

  // Abort or continue based on flag
  if (should_abort_on_error()) {
    Printf("NASan: Aborting (set NASAN_OPTIONS=abort_on_error=0 to continue)\n");
    Die();
  }
}

} // namespace __nasan
