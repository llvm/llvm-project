/// Test -m[no-]relax options.

// RUN: %clang --target=loongarch32 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=LA32
// RUN: %clang --target=loongarch64 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=LA64
// RUN: %clang --target=loongarch32 -mno-relax -S -emit-llvm %s -o - | FileCheck %s --check-prefix=LA32-NORELAX
// RUN: %clang --target=loongarch64 -mno-relax -S -emit-llvm %s -o - | FileCheck %s --check-prefix=LA64-NORELAX
// RUN: %clang --target=loongarch32 -mrelax -S -emit-llvm %s -o - | FileCheck %s --check-prefix=LA32-RELAX
// RUN: %clang --target=loongarch64 -mrelax -S -emit-llvm %s -o - | FileCheck %s --check-prefix=LA64-RELAX

/// Error when using -gsplit-dwarf with linker relaxation (-mrelax).

// RUN: %clang -### -c --target=loongarch32 -mno-relax -g -gsplit-dwarf %s 2>&1 | FileCheck %s --check-prefix=SPLIT-DWARF
// RUN: not %clang -c --target=loongarch32-linux-gnu -mrelax -gsplit-dwarf %s 2>&1 | FileCheck %s --check-prefix=ERR-SPLIT-DWARF
// RUN: not %clang -c --target=loongarch32 -mrelax -gsplit-dwarf=single %s 2>&1 | FileCheck %s --check-prefix=ERR-SPLIT-DWARF
// RUN: %clang -### -c --target=loongarch64 -mno-relax -g -gsplit-dwarf %s 2>&1 | FileCheck %s --check-prefix=SPLIT-DWARF
// RUN: not %clang -c --target=loongarch64-linux-gnu -mrelax -gsplit-dwarf %s 2>&1 | FileCheck %s --check-prefix=ERR-SPLIT-DWARF
// RUN: not %clang -c --target=loongarch64 -mrelax -gsplit-dwarf=single %s 2>&1 | FileCheck %s --check-prefix=ERR-SPLIT-DWARF

// LA32: "target-features"="+32bit"
// LA64: "target-features"="+64bit,+d,+f,+lsx,+relax,+ual"

// LA32-NORELAX: "target-features"="+32bit,-relax"
// LA64-NORELAX: "target-features"="+64bit,+d,+f,+lsx,+ual,-relax"

// LA32-RELAX: "target-features"="+32bit,+relax"
// LA64-RELAX: "target-features"="+64bit,+d,+f,+lsx,+relax,+ual"

// SPLIT-DWARF:     "-split-dwarf-file"
// ERR-SPLIT-DWARF: error: -gsplit-dwarf{{.*}} is unsupported with LoongArch linker relaxation (-mrelax)

int foo(void) {
  return 3;
}
