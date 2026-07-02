// Binaries with relative relocations targeting odd addresses that are not word
// aligned are handled correctly in the presence of a relr section.

// RUN: %clang %cflags -fPIC -pie %s -o %t -Wl,-z,pack-relative-relocs -Wl,-q

// The binary contains 2 relocations in .rela.dyn, one of them is at an odd
// offset. It contains one relocation in .relr.dyn.
//
// RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=SANITY
// SANITY:     Relocations [
// SANITY:       Section {{.*}} .rela.dyn {
// SANITY-DAG:     0x{{[0-9A-F]*[02468ACE]}} R_X86_64_RELATIVE
// SANITY-DAG:     0x{{[0-9A-F]*[13579BDF]}} R_X86_64_RELATIVE
// SANITY:       }
// SANITY:       Section {{.*}} .relr.dyn {
// SANITY:         R_X86_64_RELATIVE
// SANITY:       }

// Rewrite the binary with BOLT.
//
// RUN: llvm-bolt %t -o %t.bolt
// RUN: llvm-readobj -r %t.bolt | FileCheck %s --check-prefix=SANITY

struct __attribute__((packed)) S {
  const char *Ptr;
  char Pad;
};

/// Ends up in .data.rel.ro with section alignment 1. lld cannot turn these into
/// relr relocations for two reasons: one of the relocations will be at an odd
/// offset, which cannot be represented with a relr relocation. BOLT has to emit
/// this one as rela. The other relocation will be at an even offset. lld
/// (currently) only promotes rela to relr for input sections with alignment
/// >= 2. BOLT can re-emit this one either as rela or promote it to an relr.
/// However, BOLT (currently) cannot grow the relr section, which forces
/// emissions as rela.
__attribute__((aligned(1))) const struct S RO[] = {{"s1", 1}, {"s2", 2}};

/// Ends up in .data with section alignment 8. Emits relr relocation.
static void *RW = &RW;

int main() { return (long)RO[0].Ptr + (long)RO[1].Ptr + (long)RW; }
