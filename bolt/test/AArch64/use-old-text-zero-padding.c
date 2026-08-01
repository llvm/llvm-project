// Verify that BOLT will zero pad section end when --use-old-text
// is specified, such that there won't be stale instructions left
// there from input library.

// RUN: rm -rf %t && split-file %s %t

// Use a linker script to force section ordering .text -> .rodata so
// the test can locate the padding immediately after .text.

// RUN: %clang %cflags -falign-functions=64 \
// RUN:   -Wl,--script=%t/script.ld %t/test.c -o %t/test -Wl,-q
// RUN: llvm-bolt %t/test -o %t/test.bolt --use-old-text --align-text=4

// RUN: llvm-objdump -s -j .rodata %t/test.bolt \
// RUN:   | FileCheck %s --check-prefix=RODATA

// RODATA:     55555555 aaaaaaaa 33333333 cccccccc

// Input was built with -falign-functions=64; BOLT writes a more compact
// .text (--align-text=4) into the old region, leaving unused file space
// between the new .text end and the next section. Verify those gap bytes
// are zeroed. Padding starts at end-of-.text file offset = sh_offset +
// sh_size (read from llvm-readelf section headers).

// RUN: llvm-readelf -S %t/test.bolt | awk '$3==".text"{print $6, $7}' \
// RUN:   > %t/txt-loc
// RUN: bash -c "read O S < %t/txt-loc; \
// RUN:   od -A x -t x1 -N 32 -j \$((0x\$O + 0x\$S)) %t/test.bolt" \
// RUN:   | FileCheck %s --check-prefix=PADDING

// PADDING: {{[0-9a-f]+}} 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00

//--- script.ld
SECTIONS {
  .rodata : { *(.rodata) *(.rodata.*) }
}
INSERT AFTER .text;

//--- test.c
__attribute__((used))
const unsigned data[] = {0x55555555, 0xaaaaaaaa, 0x33333333, 0xcccccccc};

__attribute__((used, noinline)) int foo(int x, int y) { return x + y; }
__attribute__((used, noinline)) int bar(int x, int y) { return x ^ y; }

int _start(void) { return foo(1, 2) + bar(3, 4); }
