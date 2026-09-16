// REQUIRES: llvm-driver, lld, aarch64-registered-target
// REQUIRES: webassembly-registered-target

// Clang and LLD are registered in one ToolSession. Verify that frontend and
// linker jobs can execute in-process for both WebAssembly and native targets.
// RUN: split-file %s %t
// RUN: %clang --target=wasm32-unknown-unknown -nostdlib -fuse-ld=lld \
// RUN:   -Wl,--no-entry -### %t/add.c %t/sub.c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PRINT
// PRINT-COUNT-3: (in-process)
// PRINT: {{.*}}wasm-ld

// The mechanism is target-independent: a registered ELF linker is also
// dispatched in-process.
// RUN: %clang --target=aarch64-unknown-linux-gnu -nostdlib -fuse-ld=lld \
// RUN:   -Wl,-e,add -### %t/add.c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ELF-PRINT
// ELF-PRINT-COUNT-2: (in-process)
// ELF-PRINT: {{.*}}ld.lld

// An unrelated external linker with a registered name remains out-of-process.
// RUN: %if !system-windows %{ mkdir -p %t.external && touch %t.external/ld && \
// RUN:   chmod +x %t.external/ld %}
// RUN: %if !system-windows %{ %clang --target=aarch64-unknown-linux-gnu \
// RUN:   -nostdlib -B%t.external -### %t/add.c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EXTERNAL %}
// EXTERNAL: (in-process)
// EXTERNAL-NOT: (in-process)
// EXTERNAL: {{.*}}external{{[/\\]}}ld

// RUN: env LLD_IN_TEST=2 %clang --target=aarch64-unknown-linux-gnu \
// RUN:   -nostdlib -fuse-ld=lld -Wl,-e,add %t/add.c -o %t.elf
// RUN: llvm-readobj --file-headers %t.elf \
// RUN:   | FileCheck %s --check-prefix=ELF
// ELF: Format: elf64-littleaarch64

// RUN: env LLD_IN_TEST=2 %clang --target=wasm32-unknown-unknown \
// RUN:   -nostdlib -fuse-ld=lld \
// RUN:   -Wl,--no-entry -Wl,--export=add -Wl,--export=sub \
// RUN:   %t/add.c %t/sub.c -o %t.wasm
// RUN: llvm-readobj --file-headers %t.wasm | FileCheck %s --check-prefix=WASM
// WASM: Format: WASM

//--- add.c
int add(int lhs, int rhs) { return lhs + rhs; }

//--- sub.c
int sub(int lhs, int rhs) { return lhs - rhs; }
