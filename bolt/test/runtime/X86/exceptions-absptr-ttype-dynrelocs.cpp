// RUN: %clangxx %cflags -O1 -fno-pic -flto -fuse-ld=lld -Wl,-q -Wl,-z,notext \
// RUN:   -Wl,-plugin-opt=-large-eh-encoding %s -o %t.exe
// RUN: %t.exe
// RUN: llvm-bolt %t.exe -o %t.bolt --reorder-blocks=ext-tsp
// RUN: %t.bolt

/// Splitting must not spread the EH ranges of such a function over several
/// fragments: each fragment emits its own copy of the type table and would
/// need its own relocations, which .rela.dyn has no room for.
// RUN: llvm-bolt %t.exe -o %t.split.bolt --reorder-blocks=ext-tsp \
// RUN:   --split-functions --split-strategy=all --split-eh
// RUN: %t.split.bolt

// REQUIRES: system-linux

/// Check that BOLT preserves dynamic relocations covering the LSDA type table.
///
/// With DW_EH_PE_absptr TType encoding (selected for non-PIC x86-64 under the
/// large code model, which -large-eh-encoding also opts into), each type table
/// entry is an 8-byte absolute address. When the referenced typeinfo cannot be
/// resolved at static link time -- here because -z notext lets the linker place
/// R_X86_64_64 dynamic relocations in the read-only .gcc_except_table rather
/// than resolving them -- the entry is left as zero in the file and filled in
/// by the dynamic loader at startup.
///
/// BOLT re-emits .gcc_except_table but copies those entries as plain integers,
/// so the relocations are not carried over to the new section. The loader then
/// populates the original copy, which nothing reads, while the live table stays
/// zero. __gxx_personality_v0 treats a null type table entry as a catch-all, so
/// the *first* catch clause silently swallows every exception.
///
/// Before this is fixed the BOLT-processed binary exits non-zero because
/// classify() reports "bad_alloc" for a std::invalid_argument.

#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>

__attribute__((noinline)) void thrower() {
  throw std::invalid_argument("payload");
}

/// bad_alloc is deliberately the first clause: if its type table entry reads as
/// null it becomes a catch-all and captures the invalid_argument below it.
__attribute__((noinline)) const char *classify() {
  try {
    throw;
  } catch (const std::bad_alloc &) {
    return "bad_alloc";
  } catch (const std::bad_cast &) {
    return "bad_cast";
  } catch (const std::invalid_argument &) {
    return "invalid_argument";
  } catch (...) {
    return "other";
  }
}

int main() {
  try {
    thrower();
  } catch (...) {
    const char *Result = classify();
    std::printf("caught as: %s\n", Result);
    return std::strcmp(Result, "invalid_argument") == 0 ? 0 : 1;
  }
  return 2;
}
