// RUN: rm -rf %t
// RUN: split-file %s %t

// We need to check we don't waste source location space for the same file
// (i.e. base.modulemap) when it's passed to multiple PCM file.
//
// *** First, try to use normal module map files.
// RUN: %clang_cc1 -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/base.modulemap -fmodule-map-file=%t/a.modulemap \
// RUN:   -fmodule-name=a -xc++ -emit-module -o %t/a.pcm %t/a.modulemap

// RUN: %clang_cc1 -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/base.modulemap -fmodule-map-file=%t/a.modulemap -fmodule-map-file=%t/b.modulemap \
// RUN:   -fmodule-file=%t/a.pcm \
// RUN:   -fmodule-name=b -xc++ -emit-module -o %t/b.pcm %t/b.modulemap

// RUN: %clang_cc1 -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/base.modulemap -fmodule-map-file=%t/a.modulemap -fmodule-map-file=%t/b.modulemap \
// RUN:   -fmodule-file=%t/a.pcm -fmodule-file=%t/b.pcm \
// RUN:   -fsyntax-only -print-stats %t/use.cpp 2>&1 \
// RUN:      | FileCheck %t/use.cpp

// *** Switch to -flate-module-map-file and check it produces less loaded SLO entries.
// RUN: rm %t/*.pcm

// RUN: %clang_cc1 -fmodules -fno-implicit-modules \
// RUN:   -flate-module-map-file=%t/base.modulemap -flate-module-map-file=%t/a.modulemap \
// RUN:   -fmodule-name=a -xc++ -emit-module -o %t/a.pcm %t/a.modulemap

// RUN: %clang_cc1 -fmodules -fno-implicit-modules \
// RUN:   -flate-module-map-file=%t/base.modulemap -flate-module-map-file=%t/a.modulemap -flate-module-map-file=%t/b.modulemap \
// RUN:   -fmodule-file=%t/a.pcm \
// RUN:   -fmodule-name=b -xc++ -emit-module -o %t/b.pcm %t/b.modulemap

// RUN: %clang_cc1 -fmodules -fno-implicit-modules \
// RUN:   -flate-module-map-file=%t/base.modulemap -flate-module-map-file=%t/a.modulemap -flate-module-map-file=%t/b.modulemap \
// RUN:   -fmodule-file=%t/a.pcm -fmodule-file=%t/b.pcm \
// RUN:   -fsyntax-only -print-stats %t/use.cpp 2>&1 \
// RUN:      | FileCheck --check-prefix=CHECK-LATE %t/use.cpp

//--- use.cpp
// This is a very shaky check, it would be nice if it was more directly looking at which files were loaded.
// We load 2 SLocEntries less with flate-module-map-file, they correspond to savings from base.modulemap and a.modulemap
// reusing the FileID in a.pcm and b.pcm.
// We also have 3 less local SLocEntries, because we get to reuse FileIDs all module maps from PCMs.
//
// CHECK: Source Manager Stats:
// CHECK: 7 local SLocEntries
// CHECK: 13 loaded SLocEntries

// CHECK-LATE: Source Manager Stats:
// CHECK-LATE: 4 local SLocEntries
// CHECK-LATE: 11 loaded SLocEntries
#include "a.h"
#include "b.h"
#include "assert.h"

int main() {
    return a() + b();
}

//--- base.modulemap
module "base" {
    textual header "assert.h"
}

//--- a.modulemap
module "a" {
    header "a.h"
    use "base"
}

//--- b.modulemap
module "b" {
    header "b.h"
    use "a"
    use "base"
}


//--- assert.h
#define ASSERT

//--- a.h
#include "assert.h"
inline int a() { return 1; }

//--- b.h
#include "a.h"
#include "assert.h"

inline int b() { return a() + 1; }

