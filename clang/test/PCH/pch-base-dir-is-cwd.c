// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: mkdir %t/original/pch

// The PCH option does not change the base directory of a module file.
// RUN: cd %t/original && %clang_cc1 -emit-module -x c modules/module.modulemap -fmodule-name=ExplicitTypes -o pch/isolated.pcm -fmodules -fno-implicit-modules -fpch-base-dir-is-cwd
// RUN: llvm-bcanalyzer -dump -disable-histogram %t/original/pch/isolated.pcm | FileCheck %s --check-prefix=MODULE
// MODULE: <MODULE_DIRECTORY{{.*}}> blob data = '{{(/|[A-Za-z]:).*[/\\]}}original{{[/\\]}}modules'
// MODULE: <INPUT_FILE{{.*}}> blob data = 'explicit.h'

// Default PCH generation still stores absolute input paths.
// RUN: cd %t/original && %clang_cc1 -emit-pch -x c-header forced.h -o pch/default.pch
// RUN: llvm-bcanalyzer -dump -disable-histogram %t/original/pch/default.pch | FileCheck %s --check-prefix=DEFAULT
// DEFAULT: <ORIGINAL_FILE{{.*}}> blob data = '{{(/|[A-Za-z]:).*[/\\]}}original{{[/\\]}}forced.h'
// DEFAULT: <INPUT_FILE{{.*}}> blob data = '{{(/|[A-Za-z]:).*[/\\]}}original{{[/\\]}}forced.h'

// Paths under the cwd are relative; paths outside it remain absolute.
// RUN: cd %t/original && %clang_cc1 -emit-module -x c modules/module.modulemap -fmodule-name=ExplicitTypes -o pch/ExplicitTypes.pcm -fmodules -fno-implicit-modules -fmodule-file-home-is-cwd
// RUN: cd %t/original && %clang_cc1 -emit-pch -x c-header headers/bridge.h -o pch/bridge.pch -fpch-base-dir-is-cwd -isysroot %t/original -fmodules -fno-implicit-modules -fmodule-map-file=modules/module.modulemap -fmodule-file=ExplicitTypes=pch/ExplicitTypes.pcm -I modules -I %t/shared -include ./forced.h -imacros ./macros.h
// RUN: llvm-bcanalyzer -dump -disable-histogram -show-binary-blobs %t/original/pch/bridge.pch > %t/bridge.dump
// RUN: FileCheck %s --check-prefix=PATHS --input-file=%t/bridge.dump
// PATHS: <IMPORT {{.*}}> blob data = 'ExplicitTypes{{.*}}pch{{[/\\]+}}ExplicitTypes.pcm'
// PATHS: <ORIGINAL_FILE{{.*}}> blob data = 'headers{{[/\\]+}}bridge.h'
// PATHS-DAG: <INPUT_FILE{{.*}}> blob data = 'forced.h'
// PATHS-DAG: <INPUT_FILE{{.*}}> blob data = 'headers{{[/\\]+}}bridge.h'
// PATHS-DAG: <INPUT_FILE{{.*}}> blob data = 'macros.h'
// PATHS-DAG: <INPUT_FILE{{.*}}> blob data = '{{(/|[A-Za-z]:).*[/\\]}}shared{{[/\\]+}}types.h'

// Reuse the dump, but check it separately so the negative checks cover the whole
// file rather than only the gaps between positive checks. The summary footer
// includes the PCH filename, so only reject record blobs.
// RUN: FileCheck %s --check-prefix=NO-ABSOLUTE --input-file=%t/bridge.dump
// NO-ABSOLUTE-NOT: blob data = '{{.*[/\\]}}original{{[/\\]}}
// NO-ABSOLUTE-NOT: <MODULE_DIRECTORY

// The two options select different base directories, with or without a sysroot.
// RUN: not %clang_cc1 -emit-pch -x c-header %t/original/headers/bridge.h -o %t/error.pch -fpch-base-dir-is-cwd -relocatable-pch -isysroot %t/original 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: not %clang_cc1 -emit-pch -x c-header %t/original/headers/bridge.h -o %t/error.pch -fpch-base-dir-is-cwd -relocatable-pch 2>&1 | FileCheck %s --check-prefix=ERROR
// ERROR: -fpch-base-dir-is-cwd cannot be used with -relocatable-pch

// PCHs built in different working directories are bit-identical when all their
// inputs are inside those directories. Preserve input timestamps when copying.
// RUN: cd %t/original && %clang_cc1 -emit-pch -x c-header headers/identical.h -o pch/identical.pch -fpch-base-dir-is-cwd
// RUN: cp -pR %t/original %t/relocated
// RUN: cd %t/relocated && %clang_cc1 -emit-pch -x c-header headers/identical.h -o pch/identical.pch -fpch-base-dir-is-cwd
// RUN: cmp %t/original/pch/identical.pch %t/relocated/pch/identical.pch

// Consume the original PCH from the relocated tree after removing the original.
// RUN: rm -rf %t/original
// RUN: cd %t/relocated && %clang_cc1 -fsyntax-only main.c -include-pch pch/bridge.pch -fmodules -fno-implicit-modules -fmodule-map-file=modules/module.modulemap -fmodule-file=ExplicitTypes=pch/ExplicitTypes.pcm -I modules -I %t/shared -include ./forced.h -imacros ./macros.h

// PCH generation also works when the working directory does not exist.
// RUN: %clang_cc1 -emit-pch -x c-header %t/relocated/forced.h -o %t/no-cwd.pch -fpch-base-dir-is-cwd -working-directory %t/does-not-exist

// The driver forwards the flag to -cc1.
// RUN: %clang -### -x c-header %s -fpch-base-dir-is-cwd -o %t/driver.pch 2>&1 | FileCheck %s --check-prefix=DRIVER
// DRIVER: "-cc1"
// DRIVER-SAME: "-fpch-base-dir-is-cwd"

//--- original/headers/bridge.h
#include "explicit.h"
#include "types.h"

//--- original/headers/identical.h
#include "../forced.h"
#include "../macros.h"
#include "../modules/explicit.h"

//--- original/modules/module.modulemap
module ExplicitTypes { header "explicit.h" export * }

//--- original/modules/explicit.h
typedef int ExplicitValue;

//--- shared/types.h
typedef struct { int x; } BridgePoint;

//--- original/forced.h
// Leave this unguarded to catch re-including the file after loading the PCH.
typedef struct { int x; } ForcedPoint;

//--- original/macros.h
#define BRIDGE_VALUE 42

//--- original/main.c
int getX(BridgePoint point, ForcedPoint forced, ExplicitValue value) {
  return point.x + forced.x + value + BRIDGE_VALUE;
}
