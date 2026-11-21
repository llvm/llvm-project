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

// The new .text is aligned to --align-text, which can push it past the start
// of the old .text and leave unused space in front of the new code. That space
// is what is left of .bolt.org.text in the output, and it has to be zeroed.

// Build a second copy that starts .text at an address which is not a multiple
// of the --align-text used below, so BOLT has to skip past it. Functions are
// padded to 256 bytes on input, which leaves enough room for the more compact
// output plus the skipped bytes.

// RUN: %clang %cflags -falign-functions=256 -fasynchronous-unwind-tables \
// RUN:   -Wl,--section-start=.text=0x10100 %t/test.c -o %t/test.misaligned \
// RUN:   -Wl,-q
// RUN: llvm-bolt %t/test.misaligned -o %t/test.misaligned.bolt --use-old-text \
// RUN:   --align-text=512

// RUN: llvm-readelf -S %t/test.misaligned.bolt | sed 's/\[ *[0-9]*\]//' \
// RUN:   | awk '$1==".bolt.org.text"{print $4, $5}' > %t/lead-loc
// RUN: bash -c "read O S < %t/lead-loc; \
// RUN:   od -A n -v -t x1 -N \$((0x\$S)) -j \$((0x\$O)) \
// RUN:     %t/test.misaligned.bolt" \
// RUN:   > %t/lead-bytes
// RUN: FileCheck %s --check-prefix=LEADING-NONEMPTY --input-file=%t/lead-bytes
// RUN: FileCheck %s --check-prefix=LEADING --input-file=%t/lead-bytes

// LEADING-NONEMPTY: 00 00 00 00
// LEADING-NOT: {{[^ 0]}}

// --hot-functions-at-end packs the new code against the end of the old .text,
// leaving the whole front of the reused region unused. Same requirement.

// RUN: llvm-bolt %t/test -o %t/test.hfe --use-old-text --align-text=4 \
// RUN:   --hot-functions-at-end

// RUN: llvm-readelf -S %t/test.hfe | sed 's/\[ *[0-9]*\]//' \
// RUN:   | awk '$1==".bolt.org.text"{print $4, $5}' > %t/hfe-loc
// RUN: bash -c "read O S < %t/hfe-loc; \
// RUN:   od -A n -v -t x1 -N \$((0x\$S)) -j \$((0x\$O)) %t/test.hfe" \
// RUN:   > %t/hfe-bytes
// RUN: FileCheck %s --check-prefix=HOTEND-NONEMPTY --input-file=%t/hfe-bytes
// RUN: FileCheck %s --check-prefix=HOTEND --input-file=%t/hfe-bytes

// HOTEND-NONEMPTY: 00 00 00 00
// HOTEND-NOT: {{[^ 0]}}

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
