// Test generating a reproducer for a modular build where required modules are
// built explicitly as separate steps.
// REQUIRES: crash-recovery, system-darwin

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/existing.yaml.in > %t/existing.yaml
//
// RUN: c-index-test core -gen-deps-reproducer -working-dir %t \
// RUN:   -- clang-executable -c %t/reproducer.c -o %t/reproducer.o \
// RUN:      -fmodules -fmodules-cache-path=%t.modulecache \
// RUN:      -ivfsoverlay %t/existing.yaml -I /virtual | FileCheck %t/reproducer.c

// Test a failed attempt at generating a reproducer.
// RUN: not c-index-test core -gen-deps-reproducer -working-dir %t \
// RUN:   -- clang-executable -c %t/failed-reproducer.c -o %t/reproducer.o \
// RUN:      -fmodules -fmodules-cache-path=%t.modulecache 2>&1 | FileCheck %t/failed-reproducer.c

// Test the content of a reproducer script.
// RUN: c-index-test core -gen-deps-reproducer -working-dir %t -o %t/repro-content \
// RUN:   -- clang-executable -c %t/reproducer.c -o %t/reproducer.o \
// RUN:      -fmodules -fmodules-cache-path=%t.modulecache \
// RUN:      -DMACRO="\$foo" \
// RUN:      -ivfsoverlay %t/existing.yaml -I /virtual \
// RUN:      -MMD -MT dependencies -MF %t/deps.d
// RUN: FileCheck %t/script-expectations.txt --input-file %t/repro-content/reproducer.sh

// Test the content of a reproducer script with CAS enabled.
// RUN: c-index-test core -gen-deps-reproducer -working-dir %t -cas-path %t/cas -o %t/repro-cas-content \
// RUN:   -- %clang -c %t/reproducer.c -o %t/reproducer.o \
// RUN:      -fmodules -fmodules-cache-path=%t.modulecache.cas \
// RUN:      -I %t/include
// RUN: FileCheck %t/cas-script-expectations.txt --input-file %t/repro-cas-content/reproducer.sh

// Verify can reproduce the original compilations with the files captured by reproducers.
// RUN: rm -rf %t.modulecache
// RUN: rm -rf %t.modulecache.cas
// RUN: rm -rf %t/cas
// RUN: rm %t/include/modular-header.h
// RUN: cd %t/repro-content
// RUN: env CLANG=%clang bash ./reproducer.sh
// RUN: cd %t/repro-cas-content
// RUN: bash ./reproducer.sh

//--- include/modular-header.h
void fn_in_modular_header(void);

//--- include/module.modulemap
module Test { header "modular-header.h" export * }

//--- reproducer.c
// CHECK: Sources and associated run script(s) are located at:
#include <modular-header.h>

void test(void) {
  fn_in_modular_header();
}

//--- failed-reproducer.c
// CHECK: fatal error: 'non-existing-header.h' file not found
#include "non-existing-header.h"

//--- existing.yaml.in
{
   "version":0,
   "case-sensitive":"false",
   "roots":[
      {
         "name":"/virtual",
         "type":"directory",
         "contents":[
            {
               "external-contents":"DIR/include/module.modulemap",
               "name":"module.modulemap",
               "type":"file"
            },
            {
               "external-contents":"DIR/include/modular-header.h",
               "name":"modular-header.h",
               "type":"file"
            }
         ]
      }
   ]
}

//--- script-expectations.txt
CHECK: CLANG:-clang-executable
CHECK: "-dependency-file" "reproducer.cache/explicitly-built-modules/Test-{{.*}}.d"
CHECK: "-o" "reproducer.cache/reproducer.o"
CHECK: -fmodule-file=Test=reproducer.cache/explicitly-built-modules/Test-{{.*}}.pcm
Verify the reproducer VFS overlay is added before the existing overlay provided on a command line.
CHECK: -ivfsoverlay "reproducer.cache/vfs/vfs.yaml"
CHECK: "-ivfsoverlay" "{{.*}}/existing.yaml"
CHECK: MACRO=\$foo
CHECK: "-dependency-file" "reproducer.cache/deps.d"

//--- cas-script-expectations.txt
CHECK: -fcas-path "reproducer.cache/cas"
CHECK: "-fcas-include-tree" "llvmcas://
CHECK: "-fmodule-file-cache-key" "Test-{{.*}}.pcm" "llvmcas://
