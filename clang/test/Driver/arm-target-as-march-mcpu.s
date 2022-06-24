/// These tests make sure that options passed to the assembler
/// via -Wa or -Xassembler are applied correctly to assembler inputs.
/// Also we check that the same priority rules apply to compiler and
/// assembler options.
///
/// Note that the cortex-a8 is armv7-a, the cortex-a32 is armv8-a
/// and clang's default Arm architecture is armv4t.

/// Basic correctness check for how the options behave when passed to the compiler
// RUN: %clang -target arm-linux-gnueabi -### -c -march=armv7-a %s 2>&1 | \
// RUN: FileCheck --check-prefix=TRIPLE-ARMV7 %s
// RUN: %clang -target arm-linux-gnueabi -### -c -march=armv7-a+crc %s 2>&1 | \
// RUN: FileCheck --check-prefixes=TRIPLE-ARMV7,EXT-CRC %s

/// -Wa/-Xassembler doesn't apply to non assembly files
// RUN: %clang -target arm-linux-gnueabi -### -c -Wa,-march=armv7-a \
// RUN: %S/Inputs/wildcard1.c 2>&1 | FileCheck --check-prefix=TRIPLE-ARMV4 %s
// RUN: %clang -target arm-linux-gnueabi -### -c -Xassembler -march=armv7-a \
// RUN: %S/Inputs/wildcard1.c 2>&1 | FileCheck --check-prefix=TRIPLE-ARMV4 %s

/// -Wa/-Xassembler does apply to assembler input
// RUN: %clang -target arm-linux-gnueabi -### -c -Wa,-march=armv7-a %s 2>&1 | \
// RUN: FileCheck --check-prefix=TRIPLE-ARMV7 %s
// RUN: %clang -target arm-linux-gnueabi -### -c -Wa,-march=armv7-a+crc %s 2>&1 | \
// RUN: FileCheck --check-prefixes=TRIPLE-ARMV7,EXT-CRC %s
// RUN: %clang -target arm-linux-gnueabi -### -c -Xassembler -march=armv7-a %s 2>&1 | \
// RUN: FileCheck --check-prefix=TRIPLE-ARMV7 %s
// RUN: %clang -target arm-linux-gnueabi -### -c -Xassembler -march=armv7-a+crc %s 2>&1 | \
// RUN: FileCheck --check-prefixes=TRIPLE-ARMV7,EXT-CRC %s

/// Check that arch name is still canonicalised
// RUN: %clang -target arm-linux-gnueabi -### -c -Wa,-march=armv7a %s 2>&1 | \
// RUN: FileCheck --check-prefix=TRIPLE-ARMV7 %s
// RUN: %clang -target arm-linux-gnueabi -### -c -Xassembler -march=armv7 %s 2>&1 | \
// RUN: FileCheck --check-prefix=TRIPLE-ARMV7 %s

/// march to compiler and assembler, we choose the one suited to the input file type
// RUN: %clang -target arm-linux-gnueabi -### -c -march=armv8-a -Wa,-march=armv7a %s 2>&1 | \
// RUN: FileCheck --check-prefix=TRIPLE-ARMV7 %s
// RUN: %clang -target arm-linux-gnueabi -### -c -march=armv7-a -Wa,-march=armv8-a \
// RUN: %S/Inputs/wildcard1.c 2>&1 | FileCheck --check-prefix=TRIPLE-ARMV7 %s

/// mcpu to compiler and march to assembler, we use the assembler's architecture for assembly files.
/// We use the target CPU for both.
// RUN: %clang -target arm-linux-gnueabi -### -c -mcpu=cortex-a8 -Wa,-march=armv8a %s 2>&1 | \
// RUN: FileCheck --check-prefixes=TRIPLE-ARMV8,CPU-A8 %s
// RUN: %clang -target arm-linux-gnueabi -### -c -mcpu=cortex-a8 -Wa,-march=armv8-a \
// RUN: %S/Inputs/wildcard1.c 2>&1 | FileCheck --check-prefixes=TRIPLE-ARMV7,CPU-A8 %s

/// march to compiler and mcpu to assembler, we use the one that matches the file type
/// (again both get the target-cpu option either way)
// RUN: %clang -target arm-linux-gnueabi -### -c -march=armv8a -Wa,-mcpu=cortex-a8 %s 2>&1 | \
// RUN: FileCheck --check-prefixes=TRIPLE-ARMV7,CPU-A8 %s
// RUN: %clang -target arm-linux-gnueabi -### -c -march=armv8a -Wa,-mcpu=cortex-a8 \
// RUN: %S/Inputs/wildcard1.c 2>&1 | FileCheck --check-prefix=TRIPLE-ARMV8 %s

/// march and mcpu to the compiler, mcpu wins
// RUN: %clang -target arm-linux-gnueabi -### -c -mcpu=cortex-a8 -march=armv8-a %s 2>&1 | \
// RUN: FileCheck --check-prefixes=TRIPLE-ARMV7,CPU-A8 %s
/// not dependent on order
// RUN: %clang -target arm-linux-gnueabi -### -c -march=armv8-a -mcpu=cortex-a8 %s 2>&1 | \
// RUN: FileCheck --check-prefixes=TRIPLE-ARMV7,CPU-A8 %s
/// or file type
// RUN: %clang -target arm-linux-gnueabi -### -c -march=armv8a -mcpu=cortex-a8 \
// RUN: %S/Inputs/wildcard1.c 2>&1 | FileCheck --check-prefixes=TRIPLE-ARMV7,CPU-A8 %s

/// If we pass mcpu and march to the assembler then mcpu's arch wins
/// (matches the compiler behaviour)
// RUN: %clang -target arm-linux-gnueabi -### -c -Wa,-mcpu=cortex-a8 -Wa,-march=armv8-a %s 2>&1 | \
// RUN: FileCheck --check-prefixes=TRIPLE-ARMV7,CPU-A8 %s
// RUN: %clang -target arm-linux-gnueabi -### -c -Wa,-mcpu=cortex-a8,-march=armv8-a %s 2>&1 | \
// RUN: FileCheck --check-prefixes=TRIPLE-ARMV7,CPU-A8 %s
// RUN: %clang -target arm-linux-gnueabi -### -c -Xassembler -march=armv8-a -Xassembler -mcpu=cortex-a8 \
// RUN: %s 2>&1 | FileCheck --check-prefixes=TRIPLE-ARMV7,CPU-A8 %s

/// Last mcpu to assembler wins
// RUN: %clang -target arm-linux-gnueabi -### -c -Wa,-mcpu=cortex-a32,-mcpu=cortex-a8 %s 2>&1 | \
// RUN: FileCheck --check-prefixes=TRIPLE-ARMV7,CPU-A8 %s
// RUN: %clang -target arm-linux-gnueabi -### -c -Wa,-mcpu=cortex-a32 -Wa,-mcpu=cortex-a8 %s 2>&1 | \
// RUN: FileCheck --check-prefix=TRIPLE-ARMV7 --check-prefix=CPU-A8 %s
// RUN: %clang -target arm-linux-gnueabi -### -c -Xassembler -mcpu=cortex-a32 -Xassembler -mcpu=cortex-a8 \
// RUN: %s 2>&1 | FileCheck --check-prefixes=TRIPLE-ARMV7,CPU-A8 %s

/// Last mcpu to compiler wins
// RUN: %clang -target arm-linux-gnueabi -### -c -mcpu=cortex-a32 -mcpu=cortex-a8 %s 2>&1 | \
// RUN: FileCheck --check-prefixes=TRIPLE-ARMV7,CPU-A8 %s

/// Last march to assembler wins
// RUN: %clang -target arm-linux-gnueabi -### -c -Wa,-march=armv8-a,-march=armv7-a %s 2>&1 | \
// RUN: FileCheck --check-prefix=TRIPLE-ARMV7 %s
// RUN: %clang -target arm-linux-gnueabi -### -c -Wa,-march=armv8-a -Wa,-march=armv7-a %s 2>&1 | \
// RUN: FileCheck --check-prefix=TRIPLE-ARMV7 %s
// RUN: %clang -target arm-linux-gnueabi -### -c -Xassembler -march=armv8-a -Xassembler -march=armv7-a \
// RUN: %s 2>&1 | FileCheck --check-prefix=TRIPLE-ARMV7 %s

/// Last march to compiler wins
// RUN: %clang -target arm-linux-gnueabi -### -c -march=armv8-a -march=armv7-a %s 2>&1 | \
// RUN: FileCheck --check-prefix=TRIPLE-ARMV7 %s

// TRIPLE-ARMV4: "-triple" "armv4t-unknown-linux-gnueabi"
// TRIPLE-ARMV7: "-triple" "armv7-unknown-linux-gnueabi"
// TRIPLE-ARMV8: "-triple" "armv8-unknown-linux-gnueabi"
// CPU-A8: "-target-cpu" "cortex-a8"
// EXT-CRC: "-target-feature" "+crc"
