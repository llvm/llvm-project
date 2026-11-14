// Check that a module file does not write its own source location entries for
// the module map that declares its module when a loaded module file already
// kept entries for the same file: it points its locations at that copy
// instead. A module map is entered once per module file that keeps entries for
// it, so in a large build this duplication is what exhausts the source
// location space.

// RUN: rm -rf %t && split-file %s %t

// mod1 is built first and keeps its own entries. mod2 and mod3 are built with
// mod1's entries loaded, so they keep none and point at mod1's copy. Reuse
// does not chain: mod3 points at mod1 as well, not at mod2, which kept nothing
// to offer. So the module map is entered twice no matter how long the chain
// gets, instead of once more for each module in it.
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules -I%t/reuse \
// RUN:   -fmodule-map-file=%t/reuse/module.modulemap \
// RUN:   -fmodule-name=mod1 -emit-module %t/reuse/module.modulemap -o %t/mod1.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules -I%t/reuse \
// RUN:   -fmodule-map-file=%t/reuse/module.modulemap -fmodule-file=%t/mod1.pcm \
// RUN:   -fmodule-name=mod2 -emit-module %t/reuse/module.modulemap -o %t/mod2.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules -I%t/reuse \
// RUN:   -fmodule-map-file=%t/reuse/module.modulemap -fmodule-file=%t/mod2.pcm \
// RUN:   -fmodule-name=mod3 -emit-module %t/reuse/module.modulemap -o %t/mod3.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules -I%t/reuse \
// RUN:   -fmodule-map-file=%t/reuse/module.modulemap -fmodule-file=%t/mod3.pcm \
// RUN:   -fsyntax-only -verify %t/reuse/check.cc

// The input file record is written whether or not the entries are, so a
// consumer still validates the module map and rebuilds when it changes.
// RUN: llvm-bcanalyzer -dump %t/mod2.pcm | FileCheck %s --check-prefix=INPUT \
// RUN:   --implicit-check-not="SM_SLOC_FILE_ENTRY{{.*}}op2=3"
// INPUT: <INPUT_FILE {{.*}}module.modulemap

// An inferred module is named by the FileID of the module map it was uniqued
// by, and a FileID is serialized as an index into the module file's own source
// location table. Such a module map keeps its own entries, so it is entered
// three times here.
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules -I%t/inferred \
// RUN:   -fmodule-map-file=%t/inferred/module.modulemap \
// RUN:   -fmodule-name=mod1 -emit-module %t/inferred/module.modulemap -o %t/inf1.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules -I%t/inferred \
// RUN:   -fmodule-map-file=%t/inferred/module.modulemap -fmodule-file=%t/inf1.pcm \
// RUN:   -fmodule-name=mod2 -emit-module %t/inferred/module.modulemap -o %t/inf2.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules -I%t/inferred \
// RUN:   -fmodule-map-file=%t/inferred/module.modulemap -fmodule-file=%t/inf2.pcm \
// RUN:   -fsyntax-only -verify %t/inferred/check.cc

// A module map read through 'extern module' has an entry whose include
// location points into the module map that read it. Leaving out that parent
// would leave the child pointing into a different module file, so the parent
// keeps its own entries too.
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules -I%t/extern \
// RUN:   -fmodule-map-file=%t/extern/module.modulemap \
// RUN:   -fmodule-name=mod1 -emit-module %t/extern/module.modulemap -o %t/ext1.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules -I%t/extern \
// RUN:   -fmodule-map-file=%t/extern/module.modulemap -fmodule-file=%t/ext1.pcm \
// RUN:   -fmodule-name=mod2 -emit-module %t/extern/module.modulemap -o %t/ext2.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules -I%t/extern \
// RUN:   -fmodule-map-file=%t/extern/module.modulemap -fmodule-file=%t/ext2.pcm \
// RUN:   -fsyntax-only -verify %t/extern/check.cc

// Two module maps that share a name are still different files, so neither is
// reused for the other and both are entered twice.
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules -I%t/same/a -I%t/same/b \
// RUN:   -fmodule-map-file=%t/same/a/module.modulemap \
// RUN:   -fmodule-map-file=%t/same/b/module.modulemap \
// RUN:   -fmodule-name=mod1 -emit-module %t/same/a/module.modulemap -o %t/same1.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules -I%t/same/a -I%t/same/b \
// RUN:   -fmodule-map-file=%t/same/a/module.modulemap \
// RUN:   -fmodule-map-file=%t/same/b/module.modulemap -fmodule-file=%t/same1.pcm \
// RUN:   -fmodule-name=mod2 -emit-module %t/same/b/module.modulemap -o %t/same2.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules -I%t/same/a -I%t/same/b \
// RUN:   -fmodule-map-file=%t/same/a/module.modulemap \
// RUN:   -fmodule-map-file=%t/same/b/module.modulemap -fmodule-file=%t/same2.pcm \
// RUN:   -fsyntax-only -verify %t/same/b/check.cc

// The same module map can be found as a user file by one compilation and as a
// system file by another, and its characteristic is observable, so a copy that
// was read the other way round is not reused. mod1 is built with the module
// map found via -I, then mod2 with it found via -isystem.
// RUN: %clang_cc1 -xc++ -std=c++17 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/syscache -I%t/system/inc \
// RUN:   -fsyntax-only %t/system/t1.cpp
// RUN: %clang_cc1 -xc++ -std=c++17 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/syscache -isystem %t/system/inc \
// RUN:   -fsyntax-only %t/system/t2.cpp
// RUN: llvm-bcanalyzer -dump %t/syscache/*/mod2-*.pcm \
// RUN:   | FileCheck %s --check-prefix=SYSKEPT
// mod1 kept the module map as a user file (op2=3), so mod2, which sees it as
// a system file, keeps its own copy (op2=4) instead of reusing mod1's.
// SYSKEPT: <SM_SLOC_FILE_ENTRY {{.*}}op2=4

// A module whose module map was reused is still rebuilt when that module map
// changes, since its input file record was written all the same.
// RUN: %clang_cc1 -xc++ -std=c++17 -I%t/stale -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -Rmodule-build -fsyntax-only %t/stale/tu.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BUILT
// RUN: %clang_cc1 -xc++ -std=c++17 -I%t/stale -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -Rmodule-build -fsyntax-only %t/stale/tu.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHED --allow-empty
// RUN: cp %t/stale/changed.modulemap %t/stale/module.modulemap
// RUN: %clang_cc1 -xc++ -std=c++17 -I%t/stale -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -Rmodule-build -fsyntax-only %t/stale/tu.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BUILT

// BUILT: building module 'mod2'
// CACHED-NOT: building module

//--- reuse/module.modulemap
module mod1 { header "mod1.h" }
module mod2 { header "mod2.h" }
module mod3 { header "mod3.h" }

//--- reuse/mod1.h
int mod1_fn(void);

//--- reuse/mod2.h
#include "mod1.h"
int mod2_fn(void);

//--- reuse/mod3.h
#include "mod2.h"
int mod3_fn(void);

//--- reuse/check.cc
#include "mod3.h"
#pragma clang __debug sloc_usage // expected-remark {{source manager location address space usage}}
// expected-note@* {{% of available space}}

// Entered once from the command line and once from mod1, and not again for
// mod2 or mod3. The line the note is reported at also shows that a location
// in mod1's copy still resolves back to this file.
// expected-note@module.modulemap:1 {{file entered 2 times}}
// expected-note@* + {{file entered}}

//--- inferred/module.modulemap
module mod1 { header "mod1.h" }
module mod2 { umbrella "sub" module * { export * } }

//--- inferred/mod1.h
int mod1_fn(void);

//--- inferred/sub/a.h
int a_fn(void);

//--- inferred/check.cc
#include "sub/a.h"
#pragma clang __debug sloc_usage // expected-remark {{source manager location address space usage}}
// expected-note@* {{% of available space}}
// expected-note@module.modulemap:1 {{file entered 3 times}}
// expected-note@* + {{file entered}}

//--- extern/module.modulemap
module mod1 { header "mod1.h" }
module mod2 { header "mod2.h" }
extern module sub "sub.modulemap"

//--- extern/sub.modulemap
module sub { header "sub.h" }

//--- extern/mod1.h
int mod1_fn(void);

//--- extern/mod2.h
#include "mod1.h"
int mod2_fn(void);

//--- extern/sub.h
int sub_fn(void);

//--- extern/check.cc
#include "mod2.h"
#pragma clang __debug sloc_usage // expected-remark {{source manager location address space usage}}
// expected-note@* {{% of available space}}
// expected-note@module.modulemap:1 {{file entered 3 times}}
// expected-note@* + {{file entered}}

//--- same/a/module.modulemap
module mod1 { header "mod1.h" }

//--- same/a/mod1.h
int mod1_fn(void);

//--- same/b/module.modulemap
module mod2 { header "mod2.h" }

//--- same/b/mod2.h
#include "mod1.h"
int mod2_fn(void);

//--- same/b/check.cc
#include "mod2.h"
#pragma clang __debug sloc_usage // expected-remark {{source manager location address space usage}}
// expected-note@* {{% of available space}}
// expected-note@module.modulemap:1 {{file entered 2 times}}
// expected-note@../a/module.modulemap:1 {{file entered 2 times}}
// expected-note@* + {{file entered}}

// The translation units sit outside the include directory so that the headers
// are found through the search path rather than next to the including file.
//--- system/inc/module.modulemap
module mod1 { header "mod1.h" export * }
module mod2 { header "mod2.h" export * }

//--- system/inc/mod1.h
int mod1_fn(void);

//--- system/inc/mod2.h
#include "mod1.h"
int mod2_fn(void);

//--- system/t1.cpp
#include "mod1.h"

//--- system/t2.cpp
#include "mod2.h"

//--- stale/module.modulemap
module mod1 { header "mod1.h" export * }
module mod2 { header "mod2.h" export * }

//--- stale/changed.modulemap
module mod1 { header "mod1.h" export * }
module mod2 { header "mod2.h" export * }
module mod9 { header "mod1.h" }

//--- stale/mod1.h
int mod1_fn(void);

//--- stale/mod2.h
#include "mod1.h"
int mod2_fn(void);

//--- stale/tu.cpp
#include "mod2.h"
int use(void) { return mod1_fn() + mod2_fn(); }
