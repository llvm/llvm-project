// REQUIRES: x86-registered-target
// REQUIRES: powerpc-registered-target
// UNSUPPORTED: darwin

//
// Generate all the types of files we can bundle.
//
// RUN: %clang -O0 -target %itanium_abi_triple %s -E -o %t.i
// RUN: %clangxx -O0 -target %itanium_abi_triple -x c++ %s -E -o %t.ii
// RUN: %clang -O0 -target %itanium_abi_triple %s -S -emit-llvm -o %t.ll
// RUN: %clang -O0 -target %itanium_abi_triple %s -c -emit-llvm -o %t.bc
// RUN: %clang -O0 -target %itanium_abi_triple %s -S -o %t.s
// RUN: %clang -O0 -target %itanium_abi_triple %s -c -o %t.o
// RUN: %clang -O0 -target %itanium_abi_triple %s -emit-ast -o %t.ast

//
// Generate an empty file to help with the checks of empty files.
//
// RUN: touch %t.empty

//
// Generate a couple of files to bundle with.
//
// RUN: echo 'Content of device file 1' > %t.tgt1
// RUN: echo 'Content of device file 2' > %t.tgt2

//
// Check help message.
//
// RUN: clang-offload-bundler --help | FileCheck %s --check-prefix CK-HELP
// CK-HELP: {{.*}}OVERVIEW: A tool to bundle several input files of the specified type <type>
// CK-HELP: {{.*}}referring to the same source file but different targets into a single
// CK-HELP: {{.*}}one. The resulting file can also be unbundled into different files by
// CK-HELP: {{.*}}this tool if -unbundle is provided.
// CK-HELP: {{.*}}USAGE: clang-offload-bundler [options]
// CK-HELP: {{.*}}-allow-missing-bundles {{.*}}- Create empty files if bundles are missing when unbundling
// CK-HELP: {{.*}}-inputs=<string>  - [<input file>,...]
// CK-HELP: {{.*}}-list {{.*}}- List bundle IDs in the bundled file.
// CK-HELP: {{.*}}-outputs=<string> - [<output file>,...]
// CK-HELP: {{.*}}-targets=<string> - [<offload kind>-<target triple>,...]
// CK-HELP: {{.*}}-type=<string>    - Type of the files to be bundled/unbundled.
// CK-HELP: {{.*}}Current supported types are:
// CK-HELP: {{.*}}i {{.*}}- cpp-output
// CK-HELP: {{.*}}ii {{.*}}- c++-cpp-output
// CK-HELP: {{.*}}ll {{.*}}- llvm
// CK-HELP: {{.*}}bc {{.*}}- llvm-bc
// CK-HELP: {{.*}}s {{.*}}- assembler
// CK-HELP: {{.*}}o {{.*}}- object
// CK-HELP: {{.*}}gch {{.*}}- precompiled-header
// CK-HELP: {{.*}}ast {{.*}}- clang AST file
// CK-HELP: {{.*}}-unbundle {{.*}}- Unbundle bundled file into several output files.

//
// Check errors.
//
// RUN: not clang-offload-bundler -type=i -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -inputs=%t.i,%t.tgt1,%t.tgt2 -outputs=%t.bundle.i -unbundle 2>&1 | FileCheck %s --check-prefix CK-ERR1
// CK-ERR1: error: only one input file supported in unbundling mode

// RUN: not clang-offload-bundler -type=i -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -inputs=%t.i -outputs=%t.bundle.i -unbundle 2>&1 | FileCheck %s --check-prefix CK-ERR1A
// CK-ERR1A: error: number of output files and targets should match in unbundling mode

// RUN: not clang-offload-bundler -type=i -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu -inputs=%t.i,%t.tgt1,%t.tgt2 -outputs=%t.bundle.i 2>&1 | FileCheck %s --check-prefix CK-ERR2
// RUN: not clang-offload-bundler -type=i -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -inputs=%t.i,%t.tgt1 -outputs=%t.bundle.i 2>&1 | FileCheck %s --check-prefix CK-ERR2
// CK-ERR2: error: number of input files and targets should match in bundling mode

// RUN: not clang-offload-bundler -type=i -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -outputs=%t.i,%t.tgt1,%t.tgt2 -inputs=%t.bundle.i 2>&1 | FileCheck %s --check-prefix CK-ERR3
// CK-ERR3: error: only one output file supported in bundling mode

// RUN: not clang-offload-bundler -type=i -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu -outputs=%t.i,%t.tgt1,%t.tgt2 -inputs=%t.bundle.i -unbundle 2>&1 | FileCheck %s --check-prefix CK-ERR4
// RUN: not clang-offload-bundler -type=i -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -outputs=%t.i,%t.tgt1 -inputs=%t.bundle.i -unbundle 2>&1 | FileCheck %s --check-prefix CK-ERR4
// CK-ERR4: error: number of output files and targets should match in unbundling mode

// RUN: not clang-offload-bundler -type=i -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -inputs=%t.i,%t.tgt1,%t.tgt2.notexist -outputs=%t.bundle.i 2>&1 | \
// RUN:   FileCheck %s -DFILE=%t.tgt2.notexist -DMSG=%errc_ENOENT --check-prefix CK-ERR5
// RUN: not clang-offload-bundler -type=i -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -outputs=%t.i,%t.tgt1,%t.tgt2 -inputs=%t.bundle.i.notexist -unbundle 2>&1 | \
// RUN:   FileCheck %s -DFILE=%t.bundle.i.notexist -DMSG=%errc_ENOENT --check-prefix CK-ERR5
// CK-ERR5: error: '[[FILE]]': [[MSG]]

// RUN: not clang-offload-bundler -type=invalid -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -inputs=%t.i,%t.tgt1,%t.tgt2 -outputs=%t.bundle.i 2>&1 | FileCheck %s -DTYPE=invalid --check-prefix CK-ERR6
// CK-ERR6: error: '[[TYPE]]': invalid file type specified

// RUN: not clang-offload-bundler 2>&1 | FileCheck %s --check-prefix CK-ERR7
// CK-ERR7: clang-offload-bundler: for the --type option: must be specified at least once!

// RUN: not clang-offload-bundler -type=i -inputs=%t.i,%t.tgt1,%t.tgt2 2>&1 | FileCheck %s -check-prefix=CK-ERR7A
// CK-ERR7A: error: for the --outputs option: must be specified at least once!

// RUN: not clang-offload-bundler -type=i -inputs=%t.i,%t.tgt1,%t.tgt2 -outputs=%t.bundle.i 2>&1 | FileCheck %s -check-prefix=CK-ERR7B
// CK-ERR7B: error: for the --targets option: must be specified at least once!

// RUN: not clang-offload-bundler -type=i -targets=hxst-powerpcxxle-ibm-linux-gnu,openxp-pxxerpc64le-ibm-linux-gnu,xpenmp-x86_xx-pc-linux-gnu -inputs=%t.i,%t.tgt1,%t.tgt2 -outputs=%t.bundle.i 2>&1 | FileCheck %s --check-prefix CK-ERR8
// CK-ERR8: error: invalid target 'hxst-powerpcxxle-ibm-linux-gnu', unknown offloading kind 'hxst', unknown target triple 'powerpcxxle-ibm-linux-gnu'

// RUN: not clang-offload-bundler -type=i -targets=host-powerpc64le-ibm-linux-gnu,openxp-pxxerpc64le-ibm-linux-gnu,xpenmp-x86_xx-pc-linux-gnu -inputs=%t.i,%t.tgt1,%t.tgt2 -outputs=%t.bundle.i 2>&1 | FileCheck %s --check-prefix CK-ERR8A
// CK-ERR8A: error: invalid target 'openxp-pxxerpc64le-ibm-linux-gnu', unknown offloading kind 'openxp', unknown target triple 'pxxerpc64le-ibm-linux-gnu'

// RUN: not clang-offload-bundler -type=i -targets=host-powerpc64le-ibm-linux-gnu,openmp-powerpc64le-ibm-linux-gnu,xpenmp-x86_xx-pc-linux-gnu -inputs=%t.i,%t.tgt1,%t.tgt2 -outputs=%t.bundle.i 2>&1 | FileCheck %s --check-prefix CK-ERR8B
// CK-ERR8B: error: invalid target 'xpenmp-x86_xx-pc-linux-gnu', unknown offloading kind 'xpenmp', unknown target triple 'x86_xx-pc-linux-gnu'

// RUN: not clang-offload-bundler -type=i -targets=openmp-powerpc64le-linux,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -inputs=%t.i,%t.tgt1,%t.tgt2 -outputs=%t.bundle.i 2>&1 | FileCheck %s --check-prefix CK-ERR9A
// CK-ERR9A: error: expecting exactly one host target but got 0

// RUN: not clang-offload-bundler -type=i -targets=host-%itanium_abi_triple,host-%itanium_abi_triple,openmp-x86_64-pc-linux-gnu -inputs=%t.i,%t.tgt1,%t.tgt2 -outputs=%t.bundle.i 2>&1 | FileCheck %s --check-prefix CK-ERR9B
// CK-ERR9B: error: Duplicate targets are not allowed

//
// Check text bundle. This is a readable format, so we check for the format we expect to find.
//
// RUN: clang-offload-bundler -type=i -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -inputs=%t.i,%t.tgt1,%t.tgt2 -outputs=%t.bundle3.i
// RUN: clang-offload-bundler -type=ii -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -inputs=%t.ii,%t.tgt1,%t.tgt2 -outputs=%t.bundle3.ii
// RUN: clang-offload-bundler -type=ll -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -inputs=%t.ll,%t.tgt1,%t.tgt2 -outputs=%t.bundle3.ll
// RUN: clang-offload-bundler -type=s -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -inputs=%t.s,%t.tgt1,%t.tgt2 -outputs=%t.bundle3.s
// RUN: clang-offload-bundler -type=s -targets=openmp-powerpc64le-ibm-linux-gnu,host-%itanium_abi_triple,openmp-x86_64-pc-linux-gnu -inputs=%t.tgt1,%t.s,%t.tgt2 -outputs=%t.bundle3.unordered.s
// RUN: FileCheck %s --input-file %t.bundle3.i --check-prefix CK-TEXTI
// RUN: FileCheck %s --input-file %t.bundle3.ii --check-prefix CK-TEXTI
// RUN: FileCheck %s --input-file %t.bundle3.ll --check-prefix CK-TEXTLL
// RUN: FileCheck %s --input-file %t.bundle3.s --check-prefix CK-TEXTS
// RUN: FileCheck %s --input-file %t.bundle3.unordered.s --check-prefix CK-TEXTS-UNORDERED

// CK-TEXTI: // __CLANG_OFFLOAD_BUNDLE____START__ host-[[HOST:.+]]
// CK-TEXTI: int A = 0;
// CK-TEXTI: test_func(void)
// CK-TEXTI: // __CLANG_OFFLOAD_BUNDLE____END__ host-[[HOST]]
// CK-TEXTI: // __CLANG_OFFLOAD_BUNDLE____START__ openmp-powerpc64le-ibm-linux-gnu
// CK-TEXTI: Content of device file 1
// CK-TEXTI: // __CLANG_OFFLOAD_BUNDLE____END__ openmp-powerpc64le-ibm-linux-gnu
// CK-TEXTI: // __CLANG_OFFLOAD_BUNDLE____START__ openmp-x86_64-pc-linux-gnu
// CK-TEXTI: Content of device file 2
// CK-TEXTI: // __CLANG_OFFLOAD_BUNDLE____END__ openmp-x86_64-pc-linux-gnu

// CK-TEXTLL: ; __CLANG_OFFLOAD_BUNDLE____START__ host-[[HOST:.+]]
// CK-TEXTLL: @A = {{.*}}global i32 0
// CK-TEXTLL: define {{.*}}@test_func()
// CK-TEXTLL: ; __CLANG_OFFLOAD_BUNDLE____END__ host-[[HOST]]
// CK-TEXTLL: ; __CLANG_OFFLOAD_BUNDLE____START__ openmp-powerpc64le-ibm-linux-gnu
// CK-TEXTLL: Content of device file 1
// CK-TEXTLL: ; __CLANG_OFFLOAD_BUNDLE____END__ openmp-powerpc64le-ibm-linux-gnu
// CK-TEXTLL: ; __CLANG_OFFLOAD_BUNDLE____START__ openmp-x86_64-pc-linux-gnu
// CK-TEXTLL: Content of device file 2
// CK-TEXTLL: ; __CLANG_OFFLOAD_BUNDLE____END__ openmp-x86_64-pc-linux-gnu

// CK-TEXTS: # __CLANG_OFFLOAD_BUNDLE____START__ host-[[HOST:.+]]
// CK-TEXTS: .globl {{.*}}test_func
// CK-TEXTS: .globl {{.*}}A
// CK-TEXTS: # __CLANG_OFFLOAD_BUNDLE____END__ host-[[HOST]]
// CK-TEXTS: # __CLANG_OFFLOAD_BUNDLE____START__ openmp-powerpc64le-ibm-linux-gnu
// CK-TEXTS: Content of device file 1
// CK-TEXTS: # __CLANG_OFFLOAD_BUNDLE____END__ openmp-powerpc64le-ibm-linux-gnu
// CK-TEXTS: # __CLANG_OFFLOAD_BUNDLE____START__ openmp-x86_64-pc-linux-gnu
// CK-TEXTS: Content of device file 2
// CK-TEXTS: # __CLANG_OFFLOAD_BUNDLE____END__ openmp-x86_64-pc-linux-gnu

// CK-TEXTS-UNORDERED: # __CLANG_OFFLOAD_BUNDLE____START__ openmp-powerpc64le-ibm-linux-gnu
// CK-TEXTS-UNORDERED: Content of device file 1
// CK-TEXTS-UNORDERED: # __CLANG_OFFLOAD_BUNDLE____END__ openmp-powerpc64le-ibm-linux-gnu
// CK-TEXTS-UNORDERED: # __CLANG_OFFLOAD_BUNDLE____START__ host-[[HOST:.+]]
// CK-TEXTS-UNORDERED: .globl {{.*}}test_func
// CK-TEXTS-UNORDERED: .globl {{.*}}A
// CK-TEXTS-UNORDERED: # __CLANG_OFFLOAD_BUNDLE____END__ host-[[HOST]]
// CK-TEXTS-UNORDERED: # __CLANG_OFFLOAD_BUNDLE____START__ openmp-x86_64-pc-linux-gnu
// CK-TEXTS-UNORDERED: Content of device file 2
// CK-TEXTS-UNORDERED: # __CLANG_OFFLOAD_BUNDLE____END__ openmp-x86_64-pc-linux-gnu

//
// Check text unbundle. Check if we get the exact same content that we bundled before for each file.
//
// RUN: clang-offload-bundler -type=i -inputs=%t.bundle3.i -list | FileCheck -check-prefix=CKLST %s
// RUN: clang-offload-bundler -type=i -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -outputs=%t.res.i,%t.res.tgt1,%t.res.tgt2 -inputs=%t.bundle3.i -unbundle
// RUN: diff %t.i %t.res.i
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: diff %t.tgt2 %t.res.tgt2
// RUN: clang-offload-bundler -type=i -targets=openmp-powerpc64le-ibm-linux-gnu -outputs=%t.res.tgt1 -inputs=%t.bundle3.i -unbundle
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: clang-offload-bundler -type=ii -inputs=%t.bundle3.ii -list | FileCheck -check-prefix=CKLST %s
// RUN: clang-offload-bundler -type=ii -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -outputs=%t.res.ii,%t.res.tgt1,%t.res.tgt2 -inputs=%t.bundle3.ii -unbundle
// RUN: diff %t.ii %t.res.ii
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: diff %t.tgt2 %t.res.tgt2
// RUN: clang-offload-bundler -type=ii -targets=openmp-x86_64-pc-linux-gnu -outputs=%t.res.tgt2 -inputs=%t.bundle3.ii -unbundle
// RUN: diff %t.tgt2 %t.res.tgt2
// RUN: clang-offload-bundler -type=ll -inputs=%t.bundle3.ll -list | FileCheck -check-prefix=CKLST %s
// RUN: clang-offload-bundler -type=ll -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -outputs=%t.res.ll,%t.res.tgt1,%t.res.tgt2 -inputs=%t.bundle3.ll -unbundle
// RUN: diff %t.ll %t.res.ll
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: diff %t.tgt2 %t.res.tgt2
// RUN: clang-offload-bundler -type=ll -targets=openmp-powerpc64le-ibm-linux-gnu -outputs=%t.res.tgt1 -inputs=%t.bundle3.ll -unbundle
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: clang-offload-bundler -type=s -inputs=%t.bundle3.s -list | FileCheck -check-prefix=CKLST %s
// RUN: clang-offload-bundler -type=s -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -outputs=%t.res.s,%t.res.tgt1,%t.res.tgt2 -inputs=%t.bundle3.s -unbundle
// RUN: diff %t.s %t.res.s
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: diff %t.tgt2 %t.res.tgt2
// RUN: clang-offload-bundler -type=s -targets=openmp-powerpc64le-ibm-linux-gnu,host-%itanium_abi_triple,openmp-x86_64-pc-linux-gnu -outputs=%t.res.tgt1,%t.res.s,%t.res.tgt2 -inputs=%t.bundle3.s -unbundle
// RUN: diff %t.s %t.res.s
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: diff %t.tgt2 %t.res.tgt2
// RUN: clang-offload-bundler -type=s -targets=openmp-x86_64-pc-linux-gnu -outputs=%t.res.tgt2 -inputs=%t.bundle3.s -unbundle
// RUN: diff %t.tgt2 %t.res.tgt2

// Check if we can unbundle a file with no magic strings.
// RUN: clang-offload-bundler -type=s -inputs=%t.s -list | FileCheck -check-prefix=CKLST2 --allow-empty %s
// RUN: clang-offload-bundler -type=s -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -outputs=%t.res.s,%t.res.tgt1,%t.res.tgt2 -inputs=%t.s -unbundle -allow-missing-bundles
// RUN: diff %t.s %t.res.s
// RUN: diff %t.empty %t.res.tgt1
// RUN: diff %t.empty %t.res.tgt2
// RUN: clang-offload-bundler -type=s -targets=openmp-powerpc64le-ibm-linux-gnu,host-%itanium_abi_triple,openmp-x86_64-pc-linux-gnu -outputs=%t.res.tgt1,%t.res.s,%t.res.tgt2 -inputs=%t.s -unbundle -allow-missing-bundles
// RUN: diff %t.s %t.res.s
// RUN: diff %t.empty %t.res.tgt1
// RUN: diff %t.empty %t.res.tgt2

// Check that bindler prints an error if given host bundle does not exist in the fat binary.
// RUN: not clang-offload-bundler -type=s -targets=host-x86_64-xxx-linux-gnu,openmp-powerpc64le-ibm-linux-gnu -outputs=%t.res.s,%t.res.tgt1 -inputs=%t.bundle3.s -unbundle -allow-missing-bundles 2>&1 | FileCheck %s --check-prefix CK-NO-HOST-BUNDLE
// CK-NO-HOST-BUNDLE: error: Can't find bundle for the host target

//
// Check binary bundle/unbundle. The content that we have before bundling must be the same we have after unbundling.
//
// RUN: clang-offload-bundler -type=bc -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -inputs=%t.bc,%t.tgt1,%t.tgt2 -outputs=%t.bundle3.bc
// RUN: clang-offload-bundler -type=gch -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -inputs=%t.ast,%t.tgt1,%t.tgt2 -outputs=%t.bundle3.gch
// RUN: clang-offload-bundler -type=ast -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -inputs=%t.ast,%t.tgt1,%t.tgt2 -outputs=%t.bundle3.ast
// RUN: clang-offload-bundler -type=ast -targets=openmp-powerpc64le-ibm-linux-gnu,host-%itanium_abi_triple,openmp-x86_64-pc-linux-gnu -inputs=%t.tgt1,%t.ast,%t.tgt2 -outputs=%t.bundle3.unordered.ast
// RUN: clang-offload-bundler -type=bc -inputs=%t.bundle3.bc -list | FileCheck -check-prefix=CKLST %s
// RUN: clang-offload-bundler -type=bc -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -outputs=%t.res.bc,%t.res.tgt1,%t.res.tgt2 -inputs=%t.bundle3.bc -unbundle
// RUN: diff %t.bc %t.res.bc
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: diff %t.tgt2 %t.res.tgt2
// RUN: clang-offload-bundler -type=bc -targets=openmp-powerpc64le-ibm-linux-gnu -outputs=%t.res.tgt1 -inputs=%t.bundle3.bc -unbundle
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: clang-offload-bundler -type=gch -inputs=%t.bundle3.gch -list | FileCheck -check-prefix=CKLST %s
// RUN: clang-offload-bundler -type=gch -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -outputs=%t.res.gch,%t.res.tgt1,%t.res.tgt2 -inputs=%t.bundle3.gch -unbundle
// RUN: diff %t.ast %t.res.gch
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: diff %t.tgt2 %t.res.tgt2
// RUN: clang-offload-bundler -type=gch -targets=openmp-x86_64-pc-linux-gnu -outputs=%t.res.tgt2 -inputs=%t.bundle3.gch -unbundle
// RUN: diff %t.tgt2 %t.res.tgt2
// RUN: clang-offload-bundler -type=ast -inputs=%t.bundle3.ast -list | FileCheck -check-prefix=CKLST %s
// RUN: clang-offload-bundler -type=ast -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -outputs=%t.res.ast,%t.res.tgt1,%t.res.tgt2 -inputs=%t.bundle3.ast -unbundle
// RUN: diff %t.ast %t.res.ast
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: diff %t.tgt2 %t.res.tgt2
// RUN: clang-offload-bundler -type=ast -targets=openmp-powerpc64le-ibm-linux-gnu,host-%itanium_abi_triple,openmp-x86_64-pc-linux-gnu -outputs=%t.res.tgt1,%t.res.ast,%t.res.tgt2 -inputs=%t.bundle3.ast -unbundle
// RUN: diff %t.ast %t.res.ast
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: diff %t.tgt2 %t.res.tgt2
// RUN: clang-offload-bundler -type=ast -targets=openmp-powerpc64le-ibm-linux-gnu,host-%itanium_abi_triple,openmp-x86_64-pc-linux-gnu -outputs=%t.res.tgt1,%t.res.ast,%t.res.tgt2 -inputs=%t.bundle3.unordered.ast -unbundle
// RUN: diff %t.ast %t.res.ast
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: diff %t.tgt2 %t.res.tgt2
// RUN: clang-offload-bundler -type=ast -targets=openmp-powerpc64le-ibm-linux-gnu -outputs=%t.res.tgt1 -inputs=%t.bundle3.ast -unbundle
// RUN: diff %t.tgt1 %t.res.tgt1

// Check if we can unbundle a file with no magic strings.
// RUN: clang-offload-bundler -type=bc -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -outputs=%t.res.bc,%t.res.tgt1,%t.res.tgt2 -inputs=%t.bc -unbundle -allow-missing-bundles
// RUN: diff %t.bc %t.res.bc
// RUN: diff %t.empty %t.res.tgt1
// RUN: diff %t.empty %t.res.tgt2
// RUN: clang-offload-bundler -type=bc -targets=openmp-powerpc64le-ibm-linux-gnu,host-%itanium_abi_triple,openmp-x86_64-pc-linux-gnu -outputs=%t.res.tgt1,%t.res.bc,%t.res.tgt2 -inputs=%t.bc -unbundle -allow-missing-bundles
// RUN: diff %t.bc %t.res.bc
// RUN: diff %t.empty %t.res.tgt1
// RUN: diff %t.empty %t.res.tgt2

// Check that we do not have to unbundle all available bundles from the fat binary.
// RUN: clang-offload-bundler -type=ast -targets=host-%itanium_abi_triple,openmp-x86_64-pc-linux-gnu -outputs=%t.res.ast,%t.res.tgt2 -inputs=%t.bundle3.unordered.ast -unbundle
// RUN: diff %t.ast %t.res.ast
// RUN: diff %t.tgt2 %t.res.tgt2

//
// Check object bundle/unbundle. The content should be bundled into an ELF
// section (we are using a PowerPC little-endian host which uses ELF). We
// have an already bundled file to check the unbundle and do a dry run on the
// bundling as it cannot be tested in all host platforms that will run these
// tests.
//

// RUN: clang-offload-bundler -type=o -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -inputs=%t.o,%t.tgt1,%t.tgt2 -outputs=%t.bundle3.o -### 2>&1 \
// RUN: | FileCheck %s -DHOST=%itanium_abi_triple -DINOBJ1=%t.o -DINOBJ2=%t.tgt1 -DINOBJ3=%t.tgt2 -DOUTOBJ=%t.bundle3.o --check-prefix CK-OBJ-CMD
// CK-OBJ-CMD: llvm-objcopy{{(.exe)?}}" "--add-section=__CLANG_OFFLOAD_BUNDLE__host-[[HOST]]={{.*}}" "--add-section=__CLANG_OFFLOAD_BUNDLE__openmp-powerpc64le-ibm-linux-gnu=[[INOBJ2]]" "--add-section=__CLANG_OFFLOAD_BUNDLE__openmp-x86_64-pc-linux-gnu=[[INOBJ3]]" "[[INOBJ1]]" "[[TEMPOBJ:.*]]"
// CK-OBJ-CMD: llvm-objcopy{{(.exe)?}}" "--set-section-flags=__CLANG_OFFLOAD_BUNDLE__host-[[HOST]]=readonly,exclude" "--set-section-flags=__CLANG_OFFLOAD_BUNDLE__openmp-powerpc64le-ibm-linux-gnu=readonly,exclude" "--set-section-flags=__CLANG_OFFLOAD_BUNDLE__openmp-x86_64-pc-linux-gnu=readonly,exclude" "[[TEMPOBJ]]" "[[OUTOBJ]]"

// RUN: clang-offload-bundler -type=o -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -inputs=%t.o,%t.tgt1,%t.tgt2 -outputs=%t.bundle3.o
// RUN: clang-offload-bundler -type=o -inputs=%t.bundle3.o -list | FileCheck -check-prefix=CKLST %s
// RUN: clang-offload-bundler -type=o -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -outputs=%t.res.o,%t.res.tgt1,%t.res.tgt2 -inputs=%t.bundle3.o -unbundle
// RUN: diff %t.bundle3.o %t.res.o
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: diff %t.tgt2 %t.res.tgt2
// RUN: clang-offload-bundler -type=o -targets=openmp-powerpc64le-ibm-linux-gnu,host-%itanium_abi_triple,openmp-x86_64-pc-linux-gnu -outputs=%t.res.tgt1,%t.res.o,%t.res.tgt2 -inputs=%t.bundle3.o -unbundle
// RUN: diff %t.bundle3.o %t.res.o
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: diff %t.tgt2 %t.res.tgt2
// RUN: clang-offload-bundler -type=o -targets=openmp-powerpc64le-ibm-linux-gnu -outputs=%t.res.tgt1 -inputs=%t.bundle3.o -unbundle
// RUN: diff %t.tgt1 %t.res.tgt1

// Check if we can unbundle a file with no magic strings.
// RUN: clang-offload-bundler -type=o -inputs=%t.o -list | FileCheck -check-prefix=CKLST2 --allow-empty %s
// RUN: clang-offload-bundler -type=o -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -outputs=%t.res.o,%t.res.tgt1,%t.res.tgt2 -inputs=%t.o -unbundle -allow-missing-bundles
// RUN: diff %t.o %t.res.o
// RUN: diff %t.empty %t.res.tgt1
// RUN: diff %t.empty %t.res.tgt2
// RUN: clang-offload-bundler -type=o -targets=openmp-powerpc64le-ibm-linux-gnu,host-%itanium_abi_triple,openmp-x86_64-pc-linux-gnu -outputs=%t.res.tgt1,%t.res.o,%t.res.tgt2 -inputs=%t.o -unbundle -allow-missing-bundles
// RUN: diff %t.o %t.res.o
// RUN: diff %t.empty %t.res.tgt1
// RUN: diff %t.empty %t.res.tgt2

//
// Check -bundle-align option
//

// RUN: clang-offload-bundler -bundle-align=4096 -type=bc -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -inputs=%t.bc,%t.tgt1,%t.tgt2 -outputs=%t.bundle3.bc
// RUN: clang-offload-bundler -type=bc -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -outputs=%t.res.bc,%t.res.tgt1,%t.res.tgt2 -inputs=%t.bundle3.bc -unbundle
// RUN: diff %t.bc %t.res.bc
// RUN: diff %t.tgt1 %t.res.tgt1
// RUN: diff %t.tgt2 %t.res.tgt2

//
// Check error due to missing bundles
//
// RUN: clang-offload-bundler -type=bc -targets=host-%itanium_abi_triple,hip-amdgcn-amd-amdhsa-gfx900 -inputs=%t.bc,%t.tgt1 -outputs=%t.hip.bundle.bc
// RUN: not clang-offload-bundler -type=bc -inputs=%t.hip.bundle.bc -outputs=%t.tmp.bc -unbundle \
// RUN:   -targets=hip-amdgcn-amd-amdhsa-gfx906 \
// RUN:   2>&1 | FileCheck -check-prefix=MISS1 %s
// RUN: not clang-offload-bundler -type=bc -inputs=%t.hip.bundle.bc -outputs=%t.tmp.bc,%t.tmp2.bc -unbundle \
// RUN:   -targets=hip-amdgcn-amd-amdhsa-gfx906,hip-amdgcn-amd-amdhsa-gfx900 \
// RUN:   2>&1 | FileCheck -check-prefix=MISS1 %s
// MISS1: error: Can't find bundles for hip-amdgcn-amd-amdhsa-gfx906
// RUN: not clang-offload-bundler -type=bc -inputs=%t.hip.bundle.bc -outputs=%t.tmp.bc,%t.tmp2.bc -unbundle \
// RUN:   -targets=hip-amdgcn-amd-amdhsa-gfx906,hip-amdgcn-amd-amdhsa-gfx803 \
// RUN:   2>&1 | FileCheck -check-prefix=MISS2 %s
// MISS2: error: Can't find bundles for hip-amdgcn-amd-amdhsa-gfx803 and hip-amdgcn-amd-amdhsa-gfx906
// RUN: not clang-offload-bundler -type=bc -inputs=%t.hip.bundle.bc -outputs=%t.tmp.bc,%t.tmp2.bc,%t.tmp3.bc -unbundle \
// RUN:   -targets=hip-amdgcn-amd-amdhsa-gfx906,hip-amdgcn-amd-amdhsa-gfx803,hip-amdgcn-amd-amdhsa-gfx1010 \
// RUN:   2>&1 | FileCheck -check-prefix=MISS3 %s
// MISS3: error: Can't find bundles for hip-amdgcn-amd-amdhsa-gfx1010, hip-amdgcn-amd-amdhsa-gfx803, and hip-amdgcn-amd-amdhsa-gfx906

//
// Check error due to duplicate targets
//
// RUN: not clang-offload-bundler -type=bc -targets=host-%itanium_abi_triple,hip-amdgcn-amd-amdhsa-gfx900,hip-amdgcn-amd-amdhsa-gfx900 \
// RUN:   -inputs=%t.bc,%t.tgt1,%t.tgt1 -outputs=%t.hip.bundle.bc 2>&1 | FileCheck -check-prefix=DUP %s
// RUN: not clang-offload-bundler -type=bc -inputs=%t.hip.bundle.bc -outputs=%t.tmp.bc,%t.tmp2.bc -unbundle \
// RUN:   -targets=hip-amdgcn-amd-amdhsa-gfx906,hip-amdgcn-amd-amdhsa-gfx906 \
// RUN:   2>&1 | FileCheck -check-prefix=DUP %s
// DUP: error: Duplicate targets are not allowed
//
// Check -list option
//

// RUN: clang-offload-bundler -bundle-align=4096 -type=bc -targets=host-%itanium_abi_triple,openmp-powerpc64le-ibm-linux-gnu,openmp-x86_64-pc-linux-gnu -inputs=%t.bc,%t.tgt1,%t.tgt2 -outputs=%t.bundle3.bc
// RUN: not clang-offload-bundler -type=bc -inputs=%t.bundle3.bc -unbundle -list 2>&1 | FileCheck -check-prefix=CKLST-ERR %s
// CKLST-ERR: error: -unbundle and -list cannot be used together
// RUN: not clang-offload-bundler -type=bc -inputs=%t.bundle3.bc -targets=host-%itanium_abi_triple -list 2>&1 | FileCheck -check-prefix=CKLST-ERR2 %s
// CKLST-ERR2: error: -targets option is invalid for -list
// RUN: not clang-offload-bundler -type=bc -inputs=%t.bundle3.bc -outputs=out.txt -list 2>&1 | FileCheck -check-prefix=CKLST-ERR3 %s
// CKLST-ERR3: error: -outputs option is invalid for -list
// RUN: not clang-offload-bundler -type=bc -inputs=%t.bundle3.bc,%t.bc -list 2>&1 | FileCheck -check-prefix=CKLST-ERR4 %s
// CKLST-ERR4: error: only one input file supported for -list

// CKLST-DAG: host-
// CKLST-DAG: openmp-powerpc64le-ibm-linux-gnu
// CKLST-DAG: openmp-x86_64-pc-linux-gnu

// CKLST2-NOT: host-
// CKLST2-NOT: openmp-powerpc64le-ibm-linux-gnu
// CKLST2-NOT: openmp-x86_64-pc-linux-gnu

// Some code so that we can create a binary out of this file.
int A = 0;
void test_func(void) {
  ++A;
}
