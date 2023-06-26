/// Test -gsplit-dwarf and -gsplit-dwarf={split,single}.

/// Interaction with -g (-g2).
// RUN: %clang -### -c -target x86_64 -g -gsplit-dwarf %s 2>&1 | FileCheck %s --check-prefixes=NOINLINE,SPLIT
// RUN: %clang -### -c -target x86_64 -gsplit-dwarf -g %s 2>&1 | FileCheck %s --check-prefixes=NOINLINE,SPLIT
// RUN: %clang -### -c -target x86_64 -g2 -gsplit-dwarf %s 2>&1 | FileCheck %s --check-prefixes=NOINLINE,SPLIT
/// -gsplit-dwarf=split is equivalent to -gsplit-dwarf.
// RUN: %clang -### -c -target x86_64 -gsplit-dwarf=split -g %s 2>&1 | FileCheck %s --check-prefixes=NOINLINE,SPLIT

// INLINE:     "-fsplit-dwarf-inlining"
// NOINLINE-NOT: "-fsplit-dwarf-inlining"
// SPLIT-NOT:  "-dumpdir"
// SPLIT:      "-debug-info-kind=constructor"
// SPLIT-SAME: "-ggnu-pubnames"
// SPLIT-SAME: "-split-dwarf-file" "split-debug.dwo" "-split-dwarf-output" "split-debug.dwo"

// RUN: %clang -### -c -target wasm32 -gsplit-dwarf -g %s 2>&1 | FileCheck %s --check-prefix=SPLIT
// RUN: %clang -### -c -target amdgcn-amd-amdhsa -gsplit-dwarf -g %s 2>&1 | FileCheck %s --check-prefix=SPLIT

/// -gsplit-dwarf is a no-op on a non-ELF platform.
// RUN: %clang -### -c -target x86_64-apple-darwin  -gsplit-dwarf -g %s 2>&1 | FileCheck %s --check-prefix=DARWIN
// DARWIN:     "-debug-info-kind=standalone"
// DARWIN-NOT: "-split-dwarf

/// -gsplit-dwarf is a no-op if no -g is specified.
// RUN: %clang -### -c -target x86_64 -gsplit-dwarf %s 2>&1 | FileCheck %s --check-prefix=G0

/// ... unless -fthinlto-index= is specified.
// RUN: echo > %t.bc
// RUN: %clang -### -c -target x86_64 -fthinlto-index=dummy -gsplit-dwarf %t.bc 2>&1 | FileCheck %s --check-prefix=IR
// RUN: %clang -### -c -target x86_64 -gsplit-dwarf -x ir %t.bc 2>&1 | FileCheck %s --check-prefix=IR

// IR-NOT:  "-debug-info-kind=
// IR:      "-ggnu-pubnames"
// IR-SAME: "-split-dwarf-file" "{{.*}}.dwo" "-split-dwarf-output" "{{.*}}.dwo"

/// -gno-split-dwarf disables debug fission.
// RUN: %clang -### -c -target x86_64 -gsplit-dwarf -g -gno-split-dwarf %s 2>&1 | FileCheck %s --check-prefix=NOSPLIT
// RUN: %clang -### -c -target x86_64 -gsplit-dwarf=single -g -gno-split-dwarf %s 2>&1 | FileCheck %s --check-prefix=NOSPLIT
// RUN: %clang -### -c -target x86_64 -gno-split-dwarf -g -gsplit-dwarf %s 2>&1 | FileCheck %s --check-prefixes=NOINLINE,SPLIT

// NOSPLIT:     "-debug-info-kind=constructor"
// NOSPLIT-NOT: "-ggnu-pubnames"
// NOSPLIT-NOT: "-split-dwarf

/// Test -gsplit-dwarf=single.
// RUN: %clang -### -c -target x86_64 -gsplit-dwarf=single -g %s 2>&1 | FileCheck %s --check-prefix=SINGLE

// SINGLE: "-debug-info-kind=constructor"
// SINGLE: "-split-dwarf-file" "split-debug.o"
// SINGLE-NOT: "-split-dwarf-output"

// RUN: %clang -### -c -target x86_64 -gsplit-dwarf=single -g -o %tfoo.o %s 2>&1 | FileCheck %s --check-prefix=SINGLE_WITH_FILENAME

// SINGLE_WITH_FILENAME: "-split-dwarf-file" "{{.*}}foo.o"
// SINGLE_WITH_FILENAME-NOT: "-split-dwarf-output"

/// If linking is the final phase, the .dwo filename is derived from -o (if specified) or "a".
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -gsplit-dwarf -g %s -o obj/out 2>&1 | FileCheck %s --check-prefix=SPLIT_LINK
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -gsplit-dwarf -g %s 2>&1 | FileCheck %s --check-prefix=SPLIT_LINK_A

// SPLIT_LINK:      "-dumpdir" "obj/out-"
// SPLIT_LINK:      "-debug-info-kind=constructor"
// SPLIT_LINK-SAME: "-split-dwarf-file" "obj/out-split-debug.dwo" "-split-dwarf-output" "obj/out-split-debug.dwo"
// SPLIT_LINK_A:      "-dumpdir" "a-"
// SPLIT_LINK_A-SAME: "-split-dwarf-file" "a-split-debug.dwo" "-split-dwarf-output" "a-split-debug.dwo"

/// GCC special cases /dev/null (HOST_BIT_BUCKET) but not other special files like /dev/zero.
/// We don't apply special rules at all.
// RUN: %if !system-windows %{ %clang -### --target=x86_64-unknown-linux-gnu -gsplit-dwarf -g %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SPLIT_LINK_NULL %}

// SPLIT_LINK_NULL:      "-dumpdir" "/dev/null-"
// SPLIT_LINK_NULL-SAME: "-split-dwarf-output" "/dev/null-split-debug.dwo"

/// If -dumpdir is specified, use its value to derive the .dwo filename.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -gsplit-dwarf -g %s -o obj/out -dumpdir pf/x -c 2>&1 | FileCheck %s --check-prefix=DUMPDIR
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -gsplit-dwarf -g %s -o obj/out -dumpdir pf/x 2>&1 | FileCheck %s --check-prefix=DUMPDIR

// DUMPDIR:      "-dumpdir" "pf/x"
// DUMPDIR-SAME: "-split-dwarf-output" "pf/xsplit-debug.dwo"

/// -fsplit-dwarf-inlining
// RUN: %clang -### -c -target x86_64 -gsplit-dwarf=split -g -fsplit-dwarf-inlining %s 2>&1 | FileCheck %s --check-prefixes=INLINE,SPLIT

// RUN: %clang -### -c -target x86_64 -gsplit-dwarf=split -g -gno-pubnames %s 2>&1 | FileCheck %s --check-prefixes=NOPUBNAMES
// RUN: %clang -### -c -target x86_64 -gsplit-dwarf=split -g -gno-gnu-pubnames %s 2>&1 | FileCheck %s --check-prefixes=NOPUBNAMES
// NOPUBNAMES:      "-debug-info-kind=constructor"
// NOPUBNAMES-NOT:  "-ggnu-pubnames"
// NOPUBNAMES-SAME: "-split-dwarf-file" "split-debug.dwo" "-split-dwarf-output" "split-debug.dwo"

/// Invoke objcopy if not using the integrated assembler.
// RUN: %clang -### -c -target x86_64-unknown-linux-gnu -fno-integrated-as -gsplit-dwarf -g %s 2>&1 | FileCheck %s --check-prefix=OBJCOPY
// OBJCOPY:      objcopy{{(.exe)?}}" "--extract-dwo"
// OBJCOPY-NEXT: objcopy{{(.exe)?}}" "--strip-dwo"

/// ... but not for assembly output.
// RUN: %clang -### -S -target x86_64-unknown-linux-gnu -fno-integrated-as -gsplit-dwarf -g %s 2>&1 | FileCheck %s --check-prefix=NOOBJCOPY
// NOOBJCOPY-NOT: objcopy"

/// Interaction with -g0.
// RUN: %clang -### -c -target x86_64 -gsplit-dwarf -g0 -### %s 2>&1 | FileCheck %s --check-prefix=G0
// RUN: %clang -### -c -target x86_64 -gsplit-dwarf=single -g0 %s 2>&1 | FileCheck %s --check-prefix=G0
// RUN: %clang -### -c -target x86_64 -g0 -gsplit-dwarf %s 2>&1 | FileCheck %s --check-prefixes=G0
// RUN: %clang -### -c -target x86_64 -g0 -gsplit-dwarf=single %s 2>&1 | FileCheck %s --check-prefix=G0
// RUN: %clang -### -c -target x86_64 -gsplit-dwarf=single -g0 -fsplit-dwarf-inlining %s 2>&1 | FileCheck %s --check-prefix=G0

// G0-NOT: "-debug-info-kind=
// G0-NOT: "-split-dwarf-

/// Interaction with -g1 (-gmlt).
// RUN: %clang -### -S -target x86_64 -gsplit-dwarf -g1 %s 2>&1 | FileCheck %s --check-prefix=G1_WITH_SPLIT
// RUN: %clang -### -S -target x86_64 -gsplit-dwarf -g1 -fno-split-dwarf-inlining %s 2>&1 | FileCheck %s --check-prefix=G1_WITH_SPLIT
// RUN: %clang -### -S -target x86_64 -gmlt -gsplit-dwarf -fno-split-dwarf-inlining %s 2>&1 | FileCheck %s --check-prefix=G1_WITH_SPLIT

// G1_WITH_SPLIT: "-debug-info-kind=line-tables-only"
// G1_WITH_SPLIT: "-split-dwarf-file"
// G1_WITH_SPLIT: "-split-dwarf-output"

// RUN: %clang -### -S -target x86_64 -gsplit-dwarf -g1 -fsplit-dwarf-inlining %s 2>&1 | FileCheck %s --check-prefix=G1_NOSPLIT

// G1_NOSPLIT: "-debug-info-kind=line-tables-only"
// G1_NOSPLIT-NOT: "-split-dwarf-file"
// G1_NOSPLIT-NOT: "-split-dwarf-output"
