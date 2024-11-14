// On x86_64, the kernel does not provide the faulting address for dereferences
// of addresses greater than the 48-bit hardware addressable range, i.e.,
// `siginfo.si_addr` is zero in ASan's SEGV signal handler. This test checks
// that ASan does not misrepresent such cases as "NULL dereferences".

// REQUIRES: x86_64-target-arch
// RUN: %clang_asan %s -o %t
// RUN: %env_asan_opts=print_scariness=1 not %run %t 0x0000000000000000 2>&1 | FileCheck %s --check-prefixes=ZERO,HINT-PAGE0
// RUN: %env_asan_opts=print_scariness=1 not %run %t 0x0000000000000FFF 2>&1 | FileCheck %s --check-prefixes=LOW1,HINT-PAGE0
// RUN: %env_asan_opts=print_scariness=1 not %run %t 0x0000000000001000 2>&1 | FileCheck %s --check-prefixes=LOW2,HINT-NONE
// RUN: %env_asan_opts=print_scariness=1 not %run %t 0x4141414141414141 2>&1 | FileCheck %s --check-prefixes=HIGH,HINT-HIGHADDR
// RUN: %env_asan_opts=print_scariness=1 not %run %t 0xFFFFFFFFFFFFFFFF 2>&1 | FileCheck %s --check-prefixes=MAX,HINT-HIGHADDR

#include <stdint.h>
#include <stdlib.h>

int main(int argc, const char *argv[]) {
  const char *hex = argv[1];
  uint64_t *addr = (uint64_t *)strtoull(hex, NULL, 16);
  uint64_t x = *addr;  // segmentation fault
  return x;
}

// ZERO:  SEGV on unknown address 0x000000000000 (pc
// LOW1:  SEGV on unknown address 0x000000000fff (pc
// LOW2:  SEGV on unknown address 0x000000001000 (pc
// HIGH:  {{BUS|SEGV}} on unknown address (pc
// MAX:   {{BUS|SEGV}} on unknown address (pc

// HINT-PAGE0-NOT: Hint: this fault was caused by a dereference of a high value address
// HINT-PAGE0:     Hint: address points to the zero page.

// HINT-NONE-NOT:  Hint: this fault was caused by a dereference of a high value address
// HINT-NONE-NOT:  Hint: address points to the zero page.

// HINT-HIGHADDR:     Hint: this fault was caused by a dereference of a high value address
// HINT-HIGHADDR-NOT: Hint: address points to the zero page.

// ZERO:  SCARINESS: 10 (null-deref)
// LOW1:  SCARINESS: 10 (null-deref)
// LOW2:  SCARINESS: 20 (wild-addr-read)
// HIGH:  SCARINESS: {{(20 \(wild-addr-read\))|(60 \(wild-jump\))}}
// MAX:   SCARINESS: {{(20 \(wild-addr-read\))|(60 \(wild-jump\))}}

// TODO: Currently, register values are only printed on Mac.  Once this changes,
//       remove the 'TODO_' prefix in the following lines.
// TODO_HIGH,TODO_MAX: Register values:
// TODO_HIGH: = 0x4141414141414141
// TODO_MAX:  = 0xffffffffffffffff
