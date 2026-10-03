// Test that -fmodules-validate-directory-dependencies rebuilds an implicitly
// built module when a header is added to a directory it enumerated. Nothing the
// module recorded as an input file changes, only the listing of the directory.

// RUN: rm -rf %t
// RUN: split-file %s %t

// DEFINE: %{clang} = %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// DEFINE:   -fmodules-cache-path=%t/cache -I %t/include -Rmodule-build \
// DEFINE:   -Rmodule-validation %t/tu.c 2>&1

// RUN: %{clang} | FileCheck %s --check-prefix=BUILD-BOTH

// BUILD-BOTH-DAG: remark: building module 'Umbrella'
// BUILD-BOTH-DAG: remark: building module 'UmbrellaHeader'

// Nothing changed, so neither module is rebuilt.
// RUN: %{clang} -fmodules-validate-directory-dependencies \
// RUN:   | FileCheck %s --check-prefix=NO-BUILD --allow-empty

// NO-BUILD-NOT: remark: building module

// Make sure the headers added below are strictly newer than the module files.
// RUN: sleep 1
// RUN: touch %t/include/umbrella/nested/added.h
// RUN: touch %t/include/umbrella-header/added.h

// Without the flag, the added headers go unnoticed.
// RUN: %{clang} | FileCheck %s --check-prefix=NO-BUILD --allow-empty

// RUN: %{clang} -fmodules-validate-directory-dependencies \
// RUN:   | FileCheck %s --check-prefix=REBUILD

// REBUILD-DAG: remark: module 'Umbrella' is out of date because the contents of directory '{{.*}}nested' changed after it was built
// REBUILD-DAG: remark: building module 'Umbrella'
// REBUILD-DAG: remark: module 'UmbrellaHeader' is out of date because the contents of directory '{{.*}}umbrella-header' changed after it was built
// REBUILD-DAG: remark: building module 'UmbrellaHeader'

// The rebuilt modules are newer than the directories, so the rebuild converges.
// RUN: %{clang} -fmodules-validate-directory-dependencies \
// RUN:   | FileCheck %s --check-prefix=NO-BUILD --allow-empty

// RUN: %clang -### -fmodules -fmodules-validate-directory-dependencies \
// RUN:   -c %t/tu.c 2>&1 | FileCheck %s --check-prefix=DRIVER
// RUN: %clang -### -fmodules-validate-directory-dependencies -c %t/tu.c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DRIVER-NO-MODULES

// DRIVER: "-fmodules-validate-directory-dependencies"
// DRIVER-NO-MODULES-NOT: "-fmodules-validate-directory-dependencies"

//--- include/module.modulemap
module Umbrella {
  umbrella "umbrella"
}
module UmbrellaHeader {
  umbrella header "umbrella-header/UmbrellaHeader.h"
}

//--- include/umbrella/a.h
//--- include/umbrella/nested/b.h

//--- include/umbrella-header/UmbrellaHeader.h
#include "a.h"

//--- include/umbrella-header/a.h

//--- tu.c
#include "umbrella/a.h"
#include "umbrella-header/UmbrellaHeader.h"
