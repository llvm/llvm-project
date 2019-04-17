// Check that we split debug output properly
//
// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-ACTIONS < %t %s
//
// CHECK-ACTIONS: "-split-dwarf-file" "split-debug.dwo"

// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-ACTIONS < %t %s
// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf=split -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-ACTIONS < %t %s

// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf=single -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-ACTIONS-SINGLE-SPLIT < %t %s
//
// CHECK-ACTIONS-SINGLE-SPLIT: "-enable-split-dwarf=single"
// CHECK-ACTIONS-SINGLE-SPLIT: "-split-dwarf-file" "split-debug.o"

// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf=single -c -### -o %tfoo.o %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-SINGLE-SPLIT-FILENAME < %t %s
//
// CHECK-SINGLE-SPLIT-FILENAME: "-split-dwarf-file" "{{.*}}foo.o"

// RUN: %clang -target x86_64-macosx -gsplit-dwarf -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-NO-ACTIONS < %t %s
//
// CHECK-NO-ACTIONS-NOT: -split-dwarf


// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -o Bad.x -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-BAD < %t %s
//
// CHECK-BAD-NOT: "Bad.dwo"

// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-OPTION < %t %s
//
// RUN: %clang -target x86_64-pc-freebsd12 -gsplit-dwarf -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-OPTION < %t %s
//
// RUN: %clang -target amdgcn-amd-amdhsa -gsplit-dwarf -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-OPTION < %t %s
//
// CHECK-OPTION: "-split-dwarf-file" "split-debug.dwo"

// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -S -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-ASM < %t %s
//
// CHECK-ASM-NOT: objcopy

// RUN: %clang -target x86_64-unknown-linux-gnu -no-integrated-as -gsplit-dwarf -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-IAS < %t %s
//
// CHECK-IAS: objcopy

// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -gmlt -fno-split-dwarf-inlining -S -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-GMLT-WITH-SPLIT < %t %s
//
// CHECK-GMLT-WITH-SPLIT: "-enable-split-dwarf"
// CHECK-GMLT-WITH-SPLIT: "-debug-info-kind=line-tables-only"
// CHECK-GMLT-WITH-SPLIT: "-split-dwarf-file"

// RUN: %clang -target x86_64-unknown-linux-gnu -g -fno-split-dwarf-inlining -S -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-NOINLINE-WITHOUT-SPLIT < %t %s
//
// CHECK-NOINLINE-WITHOUT-SPLIT-NOT: "-enable-split-dwarf"
// CHECK-NOINLINE-WITHOUT-SPLIT: "-fno-split-dwarf-inlining"
// CHECK-NOINLINE-WITHOUT-SPLIT: "-debug-info-kind=limited"

// RUN: %clang -target x86_64-unknown-linux-gnu -gmlt -gsplit-dwarf -fno-split-dwarf-inlining -S -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-SPLIT-WITH-GMLT < %t %s
//
// CHECK-SPLIT-WITH-GMLT: "-enable-split-dwarf"
// CHECK-SPLIT-WITH-GMLT: "-debug-info-kind=limited"
// CHECK-SPLIT-WITH-GMLT: "-split-dwarf-file"

// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -fno-split-dwarf-inlining -S -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-SPLIT-WITH-NOINL < %t %s
//
// CHECK-SPLIT-WITH-NOINL: "-enable-split-dwarf"
// CHECK-SPLIT-WITH-NOINL: "-debug-info-kind=limited"
// CHECK-SPLIT-WITH-NOINL: "-split-dwarf-file"

// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -gmlt -S -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-GMLT-OVER-SPLIT < %t %s
//
// CHECK-GMLT-OVER-SPLIT-NOT: "-enable-split-dwarf"
// CHECK-GMLT-OVER-SPLIT: "-debug-info-kind=line-tables-only"
// CHECK-GMLT-OVER-SPLIT-NOT: "-split-dwarf-file"

// RUN: %clang -target x86_64-unknown-linux-gnu -gmlt -gsplit-dwarf -S -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-SPLIT-OVER-GMLT < %t %s
//
// CHECK-SPLIT-OVER-GMLT: "-enable-split-dwarf" "-debug-info-kind=limited"
// CHECK-SPLIT-OVER-GMLT: "-split-dwarf-file"

// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -g0 -fno-split-dwarf-inlining -S -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-G0-OVER-SPLIT < %t %s
//
// CHECK-G0-OVER-SPLIT-NOT: "-enable-split-dwarf"
// CHECK-G0-OVER-SPLIT-NOT: "-debug-info-kind
// CHECK-G0-OVER-SPLIT-NOT: "-split-dwarf-file"

// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -g0 -S -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-G0-OVER-SPLIT < %t %s
// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf=split -g0 -S -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-G0-OVER-SPLIT < %t %s
//
// CHECK-G0-OVER-SPLIT-NOT: "-enable-split-dwarf"
// CHECK-G0-OVER-SPLIT-NOT: "-debug-info-kind
// CHECK-G0-OVER-SPLIT-NOT: "-split-dwarf-file"

// RUN: %clang -target x86_64-unknown-linux-gnu -g0 -gsplit-dwarf -S -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-SPLIT-OVER-G0 < %t %s
// RUN: %clang -target x86_64-unknown-linux-gnu -g0 -gsplit-dwarf=split -S -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-SPLIT-OVER-G0 < %t %s
//
// CHECK-SPLIT-OVER-G0: "-enable-split-dwarf" "-debug-info-kind=limited"
// CHECK-SPLIT-OVER-G0: "-split-dwarf-file"
