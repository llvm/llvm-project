// REQUIRES: clang-driver

// RUN: %clang -### -S -fasm -fblocks -fbuiltin -fno-math-errno -fcommon -fpascal-strings -fno-blocks -fno-builtin -fmath-errno -fno-common -fno-pascal-strings -fblocks -fbuiltin -fmath-errno -fcommon -fpascal-strings -fsplit-stack %s 2>&1 | FileCheck -check-prefix=CHECK-OPTIONS1 %s
// RUN: %clang -### -S -fasm -fblocks -fbuiltin -fno-math-errno -fcommon -fpascal-strings -fno-asm -fno-blocks -fno-builtin -fmath-errno -fno-common -fno-pascal-strings -fno-show-source-location -fshort-enums -fshort-wchar %s 2>&1 | FileCheck -check-prefix=CHECK-OPTIONS2 %s

// CHECK-OPTIONS1: -split-stacks
// CHECK-OPTIONS1: -fgnu-keywords
// CHECK-OPTIONS1: -fblocks
// CHECK-OPTIONS1: -fpascal-strings

// CHECK-OPTIONS2: -fmath-errno
// CHECK-OPTIONS2: -fno-gnu-keywords
// CHECK-OPTIONS2: -fno-builtin
// CHECK-OPTIONS2: -fshort-enums
// CHECK-OPTIONS2: -fshort-wchar
// CHECK-OPTIONS2: -fno-common
// CHECK-OPTIONS2: -fno-show-source-location

// RUN: %clang -### -S -Wwrite-strings %s 2>&1 | FileCheck -check-prefix=WRITE-STRINGS1 %s
// WRITE-STRINGS1: -fconst-strings
// RUN: %clang -### -S -Wwrite-strings -Wno-write-strings %s 2>&1 | FileCheck -check-prefix=WRITE-STRINGS2 %s
// WRITE-STRINGS2-NOT: -fconst-strings
// RUN: %clang -### -S -Wwrite-strings -w %s 2>&1 | FileCheck -check-prefix=WRITE-STRINGS3 %s
// WRITE-STRINGS3-NOT: -fconst-strings

// RUN: %clang -### -x c++ -c %s 2>&1 | FileCheck -check-prefix=DEPRECATED-ON-CHECK %s
// RUN: %clang -### -x c++ -c -Wdeprecated %s 2>&1 | FileCheck -check-prefix=DEPRECATED-ON-CHECK %s
// RUN: %clang -### -x c++ -c -Wno-deprecated %s 2>&1 | FileCheck -check-prefix=DEPRECATED-OFF-CHECK %s
// RUN: %clang -### -x c++ -c -Wno-deprecated -Wdeprecated %s 2>&1 | FileCheck -check-prefix=DEPRECATED-ON-CHECK %s
// RUN: %clang -### -x c++ -c -w %s 2>&1 | FileCheck -check-prefix=DEPRECATED-ON-CHECK %s
// RUN: %clang -### -c %s 2>&1 | FileCheck -check-prefix=DEPRECATED-OFF-CHECK %s
// RUN: %clang -### -c -Wdeprecated %s 2>&1 | FileCheck -check-prefix=DEPRECATED-OFF-CHECK %s
// DEPRECATED-ON-CHECK: -fdeprecated-macro
// DEPRECATED-OFF-CHECK-NOT: -fdeprecated-macro

// RUN: %clang -### -S -ffp-contract=fast %s 2>&1 | FileCheck -check-prefix=FP-CONTRACT-FAST-CHECK %s
// RUN: %clang -### -S -ffast-math %s 2>&1 | FileCheck -check-prefix=FP-CONTRACT-FAST-CHECK %s
// RUN: %clang -### -S -ffp-contract=off %s 2>&1 | FileCheck -check-prefix=FP-CONTRACT-OFF-CHECK %s
// FP-CONTRACT-FAST-CHECK: -ffp-contract=fast
// FP-CONTRACT-OFF-CHECK: -ffp-contract=off

// RUN: %clang -### -S -funroll-loops %s 2>&1 | FileCheck -check-prefix=CHECK-UNROLL-LOOPS %s
// RUN: %clang -### -S -fno-unroll-loops %s 2>&1 | FileCheck -check-prefix=CHECK-NO-UNROLL-LOOPS %s
// RUN: %clang -### -S -fno-unroll-loops -funroll-loops %s 2>&1 | FileCheck -check-prefix=CHECK-UNROLL-LOOPS %s
// RUN: %clang -### -S -funroll-loops -fno-unroll-loops %s 2>&1 | FileCheck -check-prefix=CHECK-NO-UNROLL-LOOPS %s
// CHECK-UNROLL-LOOPS: "-funroll-loops"
// CHECK-NO-UNROLL-LOOPS: "-fno-unroll-loops"

// RUN: %clang -### -S -freroll-loops %s 2>&1 | FileCheck -check-prefix=CHECK-REROLL-LOOPS %s
// RUN: %clang -### -S -fno-reroll-loops %s 2>&1 | FileCheck -check-prefix=CHECK-NO-REROLL-LOOPS %s
// RUN: %clang -### -S -fno-reroll-loops -freroll-loops %s 2>&1 | FileCheck -check-prefix=CHECK-REROLL-LOOPS %s
// RUN: %clang -### -S -freroll-loops -fno-reroll-loops %s 2>&1 | FileCheck -check-prefix=CHECK-NO-REROLL-LOOPS %s
// CHECK-REROLL-LOOPS: "-freroll-loops"
// CHECK-NO-REROLL-LOOPS-NOT: "-freroll-loops"

// RUN: %clang -### -S -fprofile-sample-use=%S/Inputs/file.prof %s 2>&1 | FileCheck -check-prefix=CHECK-SAMPLE-PROFILE %s
// CHECK-SAMPLE-PROFILE: "-fprofile-sample-use={{.*}}/file.prof"

// RUN: %clang -### -S -fauto-profile=%S/Inputs/file.prof %s 2>&1 | FileCheck -check-prefix=CHECK-AUTO-PROFILE %s
// CHECK-AUTO-PROFILE: "-fprofile-sample-use={{.*}}/file.prof"

// RUN: %clang -### -S -fauto-profile=%S/Inputs/file.prof -fno-profile-sample-use %s 2>&1 | FileCheck -check-prefix=CHECK-NO-AUTO-PROFILE %s
// RUN: %clang -### -S -fauto-profile=%S/Inputs/file.prof -fno-auto-profile %s 2>&1 | FileCheck -check-prefix=CHECK-NO-AUTO-PROFILE %s
// CHECK-NO-AUTO-PROFILE-NOT: "-fprofile-sample-use={{.*}}/file.prof"

// RUN: %clang -### -S -fauto-profile=%S/Inputs/file.prof -fno-profile-sample-use -fauto-profile %s 2>&1 | FileCheck -check-prefix=CHECK-AUTO-PROFILE %s
// RUN: %clang -### -S -fauto-profile=%S/Inputs/file.prof -fno-auto-profile -fprofile-sample-use %s 2>&1 | FileCheck -check-prefix=CHECK-AUTO-PROFILE %s

// RUN: %clang -### -S -fprofile-arcs %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-ARCS %s
// RUN: %clang -### -S -fno-profile-arcs -fprofile-arcs %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-ARCS %s
// RUN: %clang -### -S -fno-profile-arcs %s 2>&1 | FileCheck -check-prefix=CHECK-NO-PROFILE-ARCS %s
// RUN: %clang -### -S -fprofile-arcs -fno-profile-arcs %s 2>&1 | FileCheck -check-prefix=CHECK-NO-PROFILE-ARCS %s
// CHECK-PROFILE-ARCS: "-femit-coverage-data"
// CHECK-NO-PROFILE-ARCS-NOT: "-femit-coverage-data"

// RUN: %clang -### -S -fprofile-dir=abc %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-DIR-UNUSED %s
// RUN: %clang -### -S -ftest-coverage -fprofile-dir=abc %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-DIR-UNUSED %s
// RUN: %clang -### -S -fprofile-arcs -fprofile-dir=abc %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-DIR %s
// RUN: %clang -### -S --coverage -fprofile-dir=abc %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-DIR %s
// RUN: %clang -### -S -fprofile-arcs -fno-profile-arcs -fprofile-dir=abc %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-DIR-NEITHER %s
// CHECK-PROFILE-DIR: "-coverage-data-file" "abc
// CHECK-PROFILE-DIR-UNUSED: argument unused
// CHECK-PROFILE-DIR-UNUSED-NOT: "-coverage-data-file" "abc
// CHECK-PROFILE-DIR-NEITHER-NOT: argument unused

// RUN: %clang -### -S -fprofile-generate %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-GENERATE-LLVM %s
// RUN: %clang -### -S -fprofile-instr-generate %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-GENERATE %s
// RUN: %clang -### -S -fprofile-generate=/some/dir %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-GENERATE-DIR %s
// RUN: %clang -### -S -fprofile-instr-generate=/tmp/somefile.profraw %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-GENERATE-FILE %s
// RUN: %clang -### -S -fprofile-generate -fprofile-use %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: %clang -### -S -fprofile-generate -fprofile-use=dir %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: %clang -### -S -fprofile-generate -fprofile-instr-use %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: %clang -### -S -fprofile-generate -fprofile-instr-use=file %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: %clang -### -S -fprofile-instr-generate -fprofile-use %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: %clang -### -S -fprofile-instr-generate -fprofile-use=dir %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: %clang -### -S -fprofile-instr-generate -fprofile-instr-use %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: %clang -### -S -fprofile-instr-generate -fprofile-instr-use=file %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: %clang -### -S -fprofile-instr-generate=file -fprofile-use %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: %clang -### -S -fprofile-instr-generate=file -fprofile-use=dir %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: %clang -### -S -fprofile-instr-generate=file -fprofile-instr-use %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: %clang -### -S -fprofile-instr-generate=file -fprofile-instr-use=file %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: %clang -### -S -fprofile-generate=dir -fprofile-use %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: %clang -### -S -fprofile-generate=dir -fprofile-use=dir %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: %clang -### -S -fprofile-generate=dir -fprofile-instr-use %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: %clang -### -S -fprofile-generate=dir -fprofile-instr-use=file %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: %clang -### -S -fprofile-instr-generate=file -fno-profile-instr-generate %s 2>&1 | FileCheck -check-prefix=CHECK-DISABLE-GEN %s
// RUN: %clang -### -S -fprofile-instr-generate -fprofile-generate %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GENERATE %s
// RUN: %clang -### -S -fprofile-instr-generate -fprofile-generate=file %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GENERATE %s
// RUN: %clang -### -S -fprofile-generate=dir -fno-profile-generate %s 2>&1 | FileCheck -check-prefix=CHECK-DISABLE-GEN %s
// RUN: %clang -### -S -fprofile-instr-use=file -fno-profile-instr-use %s 2>&1 | FileCheck -check-prefix=CHECK-DISABLE-USE %s
// RUN: %clang -### -S -fprofile-instr-use=file -fno-profile-use %s 2>&1 | FileCheck -check-prefix=CHECK-DISABLE-USE %s
// RUN: %clang -### -S -fprofile-use=file -fno-profile-use %s 2>&1 | FileCheck -check-prefix=CHECK-DISABLE-USE %s
// RUN: %clang -### -S -fprofile-use=file -fno-profile-instr-use %s 2>&1 | FileCheck -check-prefix=CHECK-DISABLE-USE %s
// RUN: %clang -### -S -fcoverage-mapping %s 2>&1 | FileCheck -check-prefix=CHECK-COVERAGE-AND-GEN %s
// RUN: %clang -### -S -fcoverage-mapping -fno-coverage-mapping %s 2>&1 | FileCheck -check-prefix=CHECK-DISABLE-COVERAGE %s
// RUN: %clang -### -S -fprofile-instr-generate -fcoverage-mapping -fno-coverage-mapping %s 2>&1 | FileCheck -check-prefix=CHECK-DISABLE-COVERAGE %s
// CHECK-PROFILE-GENERATE: "-fprofile-instrument=clang"
// CHECK-PROFILE-GENERATE-LLVM: "-fprofile-instrument=llvm"
// CHECK-PROFILE-GENERATE-DIR: "-fprofile-instrument-path=/some/dir{{/|\\\\}}{{.*}}"
// CHECK-PROFILE-GENERATE-FILE: "-fprofile-instrument-path=/tmp/somefile.profraw"
// CHECK-NO-MIX-GEN-USE: '{{[a-z=-]*}}' not allowed with '{{[a-z=-]*}}'
// CHECK-NO-MIX-GENERATE: '{{[a-z=-]*}}' not allowed with '{{[a-z=-]*}}'
// CHECK-DISABLE-GEN-NOT: "-fprofile-instrument=clang"
// CHECK-DISABLE-USE-NOT: "-fprofile-instr-use"
// CHECK-COVERAGE-AND-GEN: '-fcoverage-mapping' only allowed with '-fprofile-instr-generate'
// CHECK-DISABLE-COVERAGE-NOT: "-fcoverage-mapping"

// RUN: %clang -### -S -fprofile-use %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-USE %s
// RUN: %clang -### -S -fprofile-instr-use %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-USE %s
// RUN: mkdir -p %t.d/some/dir
// RUN: %clang -### -S -fprofile-use=%t.d/some/dir %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-USE-DIR %s
// RUN: %clang -### -S -fprofile-instr-use=/tmp/somefile.prof %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-USE-FILE %s
// CHECK-PROFILE-USE: "-fprofile-instrument-use-path=default.profdata"
// CHECK-PROFILE-USE-DIR: "-fprofile-instrument-use-path={{.*}}.d/some/dir{{/|\\\\}}default.profdata"
// CHECK-PROFILE-USE-FILE: "-fprofile-instrument-use-path=/tmp/somefile.prof"

// RUN: %clang -### -S -fvectorize %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -fno-vectorize -fvectorize %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -fno-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// RUN: %clang -### -S -fvectorize -fno-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// RUN: %clang -### -S -ftree-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -fno-tree-vectorize -fvectorize %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -fno-tree-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// RUN: %clang -### -S -ftree-vectorize -fno-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// RUN: %clang -### -S -O %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -O2 %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -Os %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -O3 %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -fno-vectorize -O3 %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -O1 -fvectorize %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -Ofast %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// RUN: %clang -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// RUN: %clang -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// RUN: %clang -### -S -Oz %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// CHECK-VECTORIZE: "-vectorize-loops"
// CHECK-NO-VECTORIZE-NOT: "-vectorize-loops"

// RUN: %clang -### -S -fslp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -fno-slp-vectorize -fslp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -fno-slp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// RUN: %clang -### -S -fslp-vectorize -fno-slp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// RUN: %clang -### -S -ftree-slp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -fno-tree-slp-vectorize -fslp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -fno-tree-slp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// RUN: %clang -### -S -ftree-slp-vectorize -fno-slp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// RUN: %clang -### -S -O %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -O2 %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -Os %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -Oz %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -O3 %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -fno-slp-vectorize -O3 %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -O1 -fslp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -Ofast %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// RUN: %clang -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// RUN: %clang -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// CHECK-SLP-VECTORIZE: "-vectorize-slp"
// CHECK-NO-SLP-VECTORIZE-NOT: "-vectorize-slp"

// RUN: %clang -### -S -fslp-vectorize-aggressive %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE-AGG %s
// RUN: %clang -### -S -fno-slp-vectorize-aggressive -fslp-vectorize-aggressive %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE-AGG %s
// RUN: %clang -### -S -fno-slp-vectorize-aggressive %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE-AGG %s
// RUN: %clang -### -S -fslp-vectorize-aggressive -fno-slp-vectorize-aggressive %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE-AGG %s
// CHECK-SLP-VECTORIZE-AGG: "-vectorize-slp-aggressive"
// CHECK-NO-SLP-VECTORIZE-AGG-NOT: "-vectorize-slp-aggressive"

// RUN: %clang -### -S -fextended-identifiers %s 2>&1 | FileCheck -check-prefix=CHECK-EXTENDED-IDENTIFIERS %s
// RUN: not %clang -### -S -fno-extended-identifiers %s 2>&1 | FileCheck -check-prefix=CHECK-NO-EXTENDED-IDENTIFIERS %s
// CHECK-EXTENDED-IDENTIFIERS: "-cc1"
// CHECK-EXTENDED-IDENTIFIERS-NOT: "-fextended-identifiers"
// CHECK-NO-EXTENDED-IDENTIFIERS: error: unsupported option '-fno-extended-identifiers'

// RUN: %clang -### -S -fno-pascal-strings -mpascal-strings %s 2>&1 | FileCheck -check-prefix=CHECK-M-PASCAL-STRINGS %s
// CHECK-M-PASCAL-STRINGS: "-fpascal-strings"

// RUN: %clang -### -S -fpascal-strings -mno-pascal-strings %s 2>&1 | FileCheck -check-prefix=CHECK-NO-M-PASCAL-STRINGS %s
// CHECK-NO-M-PASCAL-STRINGS-NOT: "-fpascal-strings"

// RUN: %clang -### -S -O4 %s 2>&1 | FileCheck -check-prefix=CHECK-MAX-O %s
// CHECK-MAX-O: warning: -O4 is equivalent to -O3
// CHECK-MAX-O: -O3

// RUN: %clang -S -O20 -o /dev/null %s 2>&1 | FileCheck -check-prefix=CHECK-INVALID-O %s
// CHECK-INVALID-O: warning: optimization level '-O20' is not supported; using '-O3' instead

// RUN: %clang -### -S -finput-charset=iso-8859-1 -o /dev/null %s 2>&1 | FileCheck -check-prefix=CHECK-INVALID-CHARSET %s
// CHECK-INVALID-CHARSET: error: invalid value 'iso-8859-1' in '-finput-charset=iso-8859-1'

// RUN: %clang -### -S -fexec-charset=iso-8859-1 -o /dev/null %s 2>&1 | FileCheck -check-prefix=CHECK-INVALID-INPUT-CHARSET %s
// CHECK-INVALID-INPUT-CHARSET: error: invalid value 'iso-8859-1' in '-fexec-charset=iso-8859-1'

// Test that we don't error on these.
// RUN: %clang -### -S -Werror                                                \
// RUN:     -falign-functions -falign-functions=2 -fno-align-functions        \
// RUN:     -fasynchronous-unwind-tables -fno-asynchronous-unwind-tables      \
// RUN:     -fbuiltin -fno-builtin                                            \
// RUN:     -fdiagnostics-show-location=once                                  \
// RUN:     -ffloat-store -fno-float-store                                    \
// RUN:     -feliminate-unused-debug-types -fno-eliminate-unused-debug-types  \
// RUN:     -fgcse -fno-gcse                                                  \
// RUN:     -fident -fno-ident                                                \
// RUN:     -fimplicit-templates -fno-implicit-templates                      \
// RUN:     -finput-charset=UTF-8                                             \
// RUN:     -fexec-charset=UTF-8                                             \
// RUN:     -fivopts -fno-ivopts                                              \
// RUN:     -fnon-call-exceptions -fno-non-call-exceptions                    \
// RUN:     -fpermissive -fno-permissive                                      \
// RUN:     -fdefer-pop -fno-defer-pop                                        \
// RUN:     -fprefetch-loop-arrays -fno-prefetch-loop-arrays                  \
// RUN:     -fprofile-correction -fno-profile-correction                      \
// RUN:     -fprofile-values -fno-profile-values                              \
// RUN:     -frounding-math -fno-rounding-math                                \
// RUN:     -fsee -fno-see                                                    \
// RUN:     -ftracer -fno-tracer                                              \
// RUN:     -funroll-all-loops -fno-unroll-all-loops                          \
// RUN:     -fuse-ld=gold                                                     \
// RUN:     -fno-builtin-foobar                                               \
// RUN:     -fno-builtin-strcat -fno-builtin-strcpy                           \
// RUN:     -fno-var-tracking                                                 \
// RUN:     -fno-unsigned-char                                                \
// RUN:     -fno-signed-char                                                  \
// RUN:     -fstrength-reduce -fno-strength-reduce                            \
// RUN:     -finline-limit=1000                                               \
// RUN:     -finline-limit                                                    \
// RUN:     -flto=1                                                           \
// RUN:     -falign-labels                                                    \
// RUN:     -falign-labels=100                                                \
// RUN:     -falign-loops                                                     \
// RUN:     -falign-loops=100                                                 \
// RUN:     -falign-jumps                                                     \
// RUN:     -falign-jumps=100                                                 \
// RUN:     -fexcess-precision=100                                            \
// RUN:     -fbranch-count-reg                                                \
// RUN:     -fcaller-saves                                                    \
// RUN:     -fno-default-inline -fdefault-inline                              \
// RUN:     -fgcse-after-reload                                               \
// RUN:     -fgcse-las                                                        \
// RUN:     -fgcse-sm                                                         \
// RUN:     -fipa-cp                                                          \
// RUN:     -finline-functions-called-once                                    \
// RUN:     -fmodulo-sched                                                    \
// RUN:     -fmodulo-sched-allow-regmoves                                     \
// RUN:     -fpeel-loops                                                      \
// RUN:     -frename-registers                                                \
// RUN:     -fschedule-insns2                                                 \
// RUN:     -fsingle-precision-constant                                       \
// RUN:     -ftree_loop_im                                                    \
// RUN:     -ftree_loop_ivcanon                                               \
// RUN:     -ftree_loop_linear                                                \
// RUN:     -funsafe-loop-optimizations                                       \
// RUN:     -fuse-linker-plugin                                               \
// RUN:     -fvect-cost-model                                                 \
// RUN:     -fvariable-expansion-in-unroller                                  \
// RUN:     -fweb                                                             \
// RUN:     -fwhole-program                                                   \
// RUN:     -fno-tree-dce -ftree-dce                                          \
// RUN:     -fno-tree-ter -ftree-ter                                          \
// RUN:     -fno-tree-vrp -ftree-vrp                                          \
// RUN:     -fno-delete-null-pointer-checks -fdelete-null-pointer-checks      \
// RUN:     -fno-inline-small-functions -finline-small-functions              \
// RUN:     -fno-fat-lto-objects -ffat-lto-objects                            \
// RUN:     -fno-merge-constants -fmerge-constants                            \
// RUN:     -fno-caller-saves -fcaller-saves                                  \
// RUN:     -fno-reorder-blocks -freorder-blocks                              \
// RUN:     -fno-schedule-insns2 -fschedule-insns2                            \
// RUN:     -fno-stack-check                                                  \
// RUN:     -fno-check-new -fcheck-new                                        \
// RUN:     -ffriend-injection                                                \
// RUN:     -fno-implement-inlines -fimplement-inlines                        \
// RUN:     -fstack-check                                                     \
// RUN:     -fforce-addr                                                      \
// RUN:     -malign-functions=100                                             \
// RUN:     -malign-loops=100                                                 \
// RUN:     -malign-jumps=100                                                 \
// RUN:     %s 2>&1 | FileCheck --check-prefix=IGNORE %s
// IGNORE-NOT: error: unknown argument

// Test that the warning is displayed on these.
// RUN: %clang -###                                                           \
// RUN: -finline-limit=1000                                                   \
// RUN: -finline-limit                                                        \
// RUN: -fexpensive-optimizations                                             \
// RUN: -fno-expensive-optimizations                                          \
// RUN: -fno-defer-pop                                                        \
// RUN: -fkeep-inline-functions                                               \
// RUN: -fno-keep-inline-functions                                            \
// RUN: -freorder-blocks                                                      \
// RUN: -falign-functions                                                     \
// RUN: -falign-functions=1                                                   \
// RUN: -ffloat-store                                                         \
// RUN: -fgcse                                                                \
// RUN: -fivopts                                                              \
// RUN: -fprefetch-loop-arrays                                                \
// RUN: -fprofile-correction                                                  \
// RUN: -fprofile-values                                                      \
// RUN: -frounding-math                                                       \
// RUN: -fschedule-insns                                                      \
// RUN: -fsignaling-nans                                                      \
// RUN: -fstrength-reduce                                                     \
// RUN: -ftracer                                                              \
// RUN: -funroll-all-loops                                                    \
// RUN: -funswitch-loops                                                      \
// RUN: -flto=1                                                               \
// RUN: -falign-labels                                                        \
// RUN: -falign-labels=100                                                    \
// RUN: -falign-loops                                                         \
// RUN: -falign-loops=100                                                     \
// RUN: -falign-jumps                                                         \
// RUN: -falign-jumps=100                                                     \
// RUN: -fexcess-precision=100                                                \
// RUN: -fbranch-count-reg                                                    \
// RUN: -fcaller-saves                                                        \
// RUN: -fno-default-inline                                                   \
// RUN: -fgcse-after-reload                                                   \
// RUN: -fgcse-las                                                            \
// RUN: -fgcse-sm                                                             \
// RUN: -fipa-cp                                                              \
// RUN: -finline-functions-called-once                                        \
// RUN: -fmodulo-sched                                                        \
// RUN: -fmodulo-sched-allow-regmoves                                         \
// RUN: -fpeel-loops                                                          \
// RUN: -frename-registers                                                    \
// RUN: -fschedule-insns2                                                     \
// RUN: -fsingle-precision-constant                                           \
// RUN: -ftree_loop_im                                                        \
// RUN: -ftree_loop_ivcanon                                                   \
// RUN: -ftree_loop_linear                                                    \
// RUN: -funsafe-loop-optimizations                                           \
// RUN: -fuse-linker-plugin                                                   \
// RUN: -fvect-cost-model                                                     \
// RUN: -fvariable-expansion-in-unroller                                      \
// RUN: -fweb                                                                 \
// RUN: -fwhole-program                                                       \
// RUN: -fcaller-saves                                                        \
// RUN: -freorder-blocks                                                      \
// RUN: -fdelete-null-pointer-checks                                          \
// RUN: -ffat-lto-objects                                                     \
// RUN: -fmerge-constants                                                     \
// RUN: -finline-small-functions                                              \
// RUN: -ftree-dce                                                            \
// RUN: -ftree-ter                                                            \
// RUN: -ftree-vrp                                                            \
// RUN: -fno-devirtualize                                                     \
// RUN: -fno-devirtualize-speculatively                                       \
// RUN: %s 2>&1 | FileCheck --check-prefix=CHECK-WARNING %s
// CHECK-WARNING-DAG: optimization flag '-finline-limit=1000' is not supported
// CHECK-WARNING-DAG: optimization flag '-finline-limit' is not supported
// CHECK-WARNING-DAG: optimization flag '-fexpensive-optimizations' is not supported
// CHECK-WARNING-DAG: optimization flag '-fno-expensive-optimizations' is not supported
// CHECK-WARNING-DAG: optimization flag '-fno-defer-pop' is not supported
// CHECK-WARNING-DAG: optimization flag '-fkeep-inline-functions' is not supported
// CHECK-WARNING-DAG: optimization flag '-fno-keep-inline-functions' is not supported
// CHECK-WARNING-DAG: optimization flag '-freorder-blocks' is not supported
// CHECK-WARNING-DAG: optimization flag '-falign-functions' is not supported
// CHECK-WARNING-DAG: optimization flag '-falign-functions=1' is not supported
// CHECK-WARNING-DAG: optimization flag '-ffloat-store' is not supported
// CHECK-WARNING-DAG: optimization flag '-fgcse' is not supported
// CHECK-WARNING-DAG: optimization flag '-fivopts' is not supported
// CHECK-WARNING-DAG: optimization flag '-fprefetch-loop-arrays' is not supported
// CHECK-WARNING-DAG: optimization flag '-fprofile-correction' is not supported
// CHECK-WARNING-DAG: optimization flag '-fprofile-values' is not supported
// CHECK-WARNING-DAG: optimization flag '-frounding-math' is not supported
// CHECK-WARNING-DAG: optimization flag '-fschedule-insns' is not supported
// CHECK-WARNING-DAG: optimization flag '-fsignaling-nans' is not supported
// CHECK-WARNING-DAG: optimization flag '-fstrength-reduce' is not supported
// CHECK-WARNING-DAG: optimization flag '-ftracer' is not supported
// CHECK-WARNING-DAG: optimization flag '-funroll-all-loops' is not supported
// CHECK-WARNING-DAG: optimization flag '-funswitch-loops' is not supported
// CHECK-WARNING-DAG: unsupported argument '1' to option 'flto='
// CHECK-WARNING-DAG: optimization flag '-falign-labels' is not supported
// CHECK-WARNING-DAG: optimization flag '-falign-labels=100' is not supported
// CHECK-WARNING-DAG: optimization flag '-falign-loops' is not supported
// CHECK-WARNING-DAG: optimization flag '-falign-loops=100' is not supported
// CHECK-WARNING-DAG: optimization flag '-falign-jumps' is not supported
// CHECK-WARNING-DAG: optimization flag '-falign-jumps=100' is not supported
// CHECK-WARNING-DAG: optimization flag '-fexcess-precision=100' is not supported
// CHECK-WARNING-DAG: optimization flag '-fbranch-count-reg' is not supported
// CHECK-WARNING-DAG: optimization flag '-fcaller-saves' is not supported
// CHECK-WARNING-DAG: optimization flag '-fno-default-inline' is not supported
// CHECK-WARNING-DAG: optimization flag '-fgcse-after-reload' is not supported
// CHECK-WARNING-DAG: optimization flag '-fgcse-las' is not supported
// CHECK-WARNING-DAG: optimization flag '-fgcse-sm' is not supported
// CHECK-WARNING-DAG: optimization flag '-fipa-cp' is not supported
// CHECK-WARNING-DAG: optimization flag '-finline-functions-called-once' is not supported
// CHECK-WARNING-DAG: optimization flag '-fmodulo-sched' is not supported
// CHECK-WARNING-DAG: optimization flag '-fmodulo-sched-allow-regmoves' is not supported
// CHECK-WARNING-DAG: optimization flag '-fpeel-loops' is not supported
// CHECK-WARNING-DAG: optimization flag '-frename-registers' is not supported
// CHECK-WARNING-DAG: optimization flag '-fschedule-insns2' is not supported
// CHECK-WARNING-DAG: optimization flag '-fsingle-precision-constant' is not supported
// CHECK-WARNING-DAG: optimization flag '-ftree_loop_im' is not supported
// CHECK-WARNING-DAG: optimization flag '-ftree_loop_ivcanon' is not supported
// CHECK-WARNING-DAG: optimization flag '-ftree_loop_linear' is not supported
// CHECK-WARNING-DAG: optimization flag '-funsafe-loop-optimizations' is not supported
// CHECK-WARNING-DAG: optimization flag '-fuse-linker-plugin' is not supported
// CHECK-WARNING-DAG: optimization flag '-fvect-cost-model' is not supported
// CHECK-WARNING-DAG: optimization flag '-fvariable-expansion-in-unroller' is not supported
// CHECK-WARNING-DAG: optimization flag '-fweb' is not supported
// CHECK-WARNING-DAG: optimization flag '-fwhole-program' is not supported
// CHECK-WARNING-DAG: optimization flag '-fcaller-saves' is not supported
// CHECK-WARNING-DAG: optimization flag '-freorder-blocks' is not supported
// CHECK-WARNING-DAG: optimization flag '-fdelete-null-pointer-checks' is not supported
// CHECK-WARNING-DAG: optimization flag '-ffat-lto-objects' is not supported
// CHECK-WARNING-DAG: optimization flag '-fmerge-constants' is not supported
// CHECK-WARNING-DAG: optimization flag '-finline-small-functions' is not supported
// CHECK-WARNING-DAG: optimization flag '-ftree-dce' is not supported
// CHECK-WARNING-DAG: optimization flag '-ftree-ter' is not supported
// CHECK-WARNING-DAG: optimization flag '-ftree-vrp' is not supported
// CHECK-WARNING-DAG: optimization flag '-fno-devirtualize' is not supported
// CHECK-WARNING-DAG: optimization flag '-fno-devirtualize-speculatively' is not supported

// Test that we mute the warning on these
// RUN: %clang -### -finline-limit=1000 -Wno-invalid-command-line-argument              \
// RUN:     %s 2>&1 | FileCheck --check-prefix=CHECK-NO-WARNING1 %s
// RUN: %clang -### -finline-limit -Wno-invalid-command-line-argument                   \
// RUN:     %s 2>&1 | FileCheck --check-prefix=CHECK-NO-WARNING2 %s
// RUN: %clang -### -finline-limit \
// RUN:     -Winvalid-command-line-argument -Wno-ignored-optimization-argument          \
// RUN:     %s 2>&1 | FileCheck --check-prefix=CHECK-NO-WARNING2 %s
// CHECK-NO-WARNING1-NOT: optimization flag '-finline-limit=1000' is not supported
// CHECK-NO-WARNING2-NOT: optimization flag '-finline-limit' is not supported

// Test that an ignored optimization argument only prints 1 warning,
// not both a warning about not claiming the arg, *and* about not supporting
// the arg; and that adding -Wno-ignored-optimization silences the warning.
//
// RUN: %clang -### -fprofile-correction %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-WARNING3 %s
// CHECK-NO-WARNING3: optimization flag '-fprofile-correction' is not supported
// CHECK-NO-WARNING3-NOT: argument unused
// RUN: %clang -### -fprofile-correction -Wno-ignored-optimization-argument %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-WARNING4 %s
// CHECK-NO-WARNING4-NOT: not supported
// CHECK-NO-WARNING4-NOT: argument unused

// RUN: %clang -### -S -fsigned-char %s 2>&1 | FileCheck -check-prefix=CHAR-SIGN1 %s
// CHAR-SIGN1-NOT: -fno-signed-char

// RUN: %clang -### -S -funsigned-char %s 2>&1 | FileCheck -check-prefix=CHAR-SIGN2 %s
// CHAR-SIGN2: -fno-signed-char

// RUN: %clang -### -S -fno-signed-char %s 2>&1 | FileCheck -check-prefix=CHAR-SIGN3 %s
// CHAR-SIGN3: -fno-signed-char

// RUN: %clang -### -S -fno-unsigned-char %s 2>&1 | FileCheck -check-prefix=CHAR-SIGN4 %s
// CHAR-SIGN4-NOT: -fno-signed-char

// RUN: %clang -### -fshort-wchar -fno-short-wchar %s 2>&1 | FileCheck -check-prefix=CHECK-WCHAR1 -check-prefix=DELIMITERS %s
// RUN: %clang -### -fno-short-wchar -fshort-wchar %s 2>&1 | FileCheck -check-prefix=CHECK-WCHAR2 -check-prefix=DELIMITERS %s
// Make sure we don't match the -NOT lines with the linker invocation.
// Delimiters match the start of the cc1 and the start of the linker lines
// DELIMITERS: {{^ *"}}
// CHECK-WCHAR1: -fno-short-wchar
// CHECK-WCHAR1-NOT: -fshort-wchar
// CHECK-WCHAR2: -fshort-wchar
// CHECK-WCHAR2-NOT: -fno-short-wchar
// DELIMITERS: {{^ *"}}

// RUN: %clang -### -fno-experimental-new-pass-manager -fexperimental-new-pass-manager %s 2>&1 | FileCheck --check-prefix=CHECK-PM --check-prefix=CHECK-NEW-PM %s
// RUN: %clang -### -fexperimental-new-pass-manager -fno-experimental-new-pass-manager %s 2>&1 | FileCheck --check-prefix=CHECK-PM --check-prefix=CHECK-NO-NEW-PM %s
// CHECK-PM-NOT: argument unused
// CHECK-NEW-PM: -fexperimental-new-pass-manager
// CHECK-NEW-PM-NOT: -fno-experimental-new-pass-manager
// CHECK-NO-NEW-PM: -fno-experimental-new-pass-manager
// CHECK-NO-NEW-PM-NOT: -fexperimental-new-pass-manager

// RUN: %clang -### -S -fstrict-return %s 2>&1 | FileCheck -check-prefix=CHECK-STRICT-RETURN %s
// RUN: %clang -### -S -fno-strict-return %s 2>&1 | FileCheck -check-prefix=CHECK-NO-STRICT-RETURN %s
// CHECK-STRICT-RETURN-NOT: "-fno-strict-return"
// CHECK-NO-STRICT-RETURN: "-fno-strict-return"

// RUN: %clang -### -S -fno-debug-info-for-profiling -fdebug-info-for-profiling %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-DEBUG %s
// RUN: %clang -### -S -fdebug-info-for-profiling -fno-debug-info-for-profiling %s 2>&1 | FileCheck -check-prefix=CHECK-NO-PROFILE-DEBUG %s
// CHECK-PROFILE-DEBUG: -fdebug-info-for-profiling
// CHECK-NO-PROFILE-DEBUG-NOT: -fdebug-info-for-profiling

// RUN: %clang -### -S -fallow-editor-placeholders %s 2>&1 | FileCheck -check-prefix=CHECK-ALLOW-PLACEHOLDERS %s
// RUN: %clang -### -S -fno-allow-editor-placeholders %s 2>&1 | FileCheck -check-prefix=CHECK-NO-ALLOW-PLACEHOLDERS %s
// CHECK-ALLOW-PLACEHOLDERS: -fallow-editor-placeholders
// CHECK-NO-ALLOW-PLACEHOLDERS-NOT: -fallow-editor-placeholders
