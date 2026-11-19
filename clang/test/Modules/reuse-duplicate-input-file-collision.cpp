// Check that two different files that agree on size and on base name are not
// reused for one another. The size is only a bucket, so the confirming
// identity check is what has to tell them apart.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/mods.map -I%t/a \
// RUN:   -fmodule-name=mod1 -emit-module %t/mods.map -o %t/mod1.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/mods.map -fmodule-file=%t/mod1.pcm -I%t/b \
// RUN:   -fmodule-name=mod2 -emit-module %t/mods.map -o %t/mod2.pcm

// RUN: llvm-bcanalyzer -dump %t/mod1.pcm > %t/mod1.txt
// RUN: llvm-bcanalyzer -dump %t/mod2.pcm > %t/mod2.txt
// RUN: cat %t/mod1.txt %t/mod2.txt | FileCheck %s

// The second record has to repeat the size the first one recorded, so the two
// headers really do share a bucket and the test cannot pass for the wrong
// reason. Neither module may reuse the other's copy, so both write their own
// entries and the SLoc index stays nonzero.

// CHECK: <INPUT_FILE {{.*}}op1=[[SIZE:[0-9]+]] {{.*}}op7={{[1-9][0-9]*}} {{.*}}a{{[/\\]}}shared.h
// CHECK: <INPUT_FILE {{.*}}op1=[[SIZE]] {{.*}}op7={{[1-9][0-9]*}} {{.*}}b{{[/\\]}}shared.h

//--- mods.map
module mod1 { header "mod1.h" export * }
module mod2 { header "mod2.h" export * }

//--- a/shared.h
#ifndef A_SHARED_H
#define A_SHARED_H
struct Collide { int aaa; };
int collide_fn_a(void);
#endif

//--- b/shared.h
#ifndef B_SHARED_H
#define B_SHARED_H
struct Collide { int bbb; };
int collide_fn_b(void);
#endif

//--- mod1.h
#include "shared.h"
int mod1_fn(void);

//--- mod2.h
#include "shared.h"
int mod2_fn(void);
