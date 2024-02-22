// REQUIRES: shell

// RUN: rm -rf %t && mkdir -p %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-include-tree-full \
// RUN:   -cas-path %t/cas -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext \
// RUN:   -fcas-plugin-option upstream-path=%t/cas-upstream -fcas-plugin-option no-logging \
// RUN:   > %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/cc1.rsp

// RUN: (cd %t; %clang @%t/cc1.rsp)
// RUN: (cd %t; %clang @%t/cc1.rsp -Rcompile-job-cache-hit \
// RUN:   -serialize-diagnostic-file %t/t1.dia 2> %t/output1.txt)

// Verify the warning was recorded and we compare populated .dia files.
// RUN: c-index-test -read-diagnostics %t/t1.dia 2>&1 | FileCheck %s --check-prefix=DIAGS
// DIAGS: warning: some warning

// RUN: cat %t/output1.txt | grep llvmcas | sed \
// RUN:   -e "s/^.*hit for '//" \
// RUN:   -e "s/' .*$//" > %t/cache-key

// Delete the "local" cache and use the "upstream" one to re-materialize the outputs locally.
// RUN: rm -rf %t/cas

// Re-run the scan to populate the include-tree in the cas
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-include-tree-full \
// RUN:   -cas-path %t/cas -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext \
// RUN:   -fcas-plugin-option upstream-path=%t/cas-upstream -fcas-plugin-option no-logging \
// RUN:   > %t/deps2.json
// RUN: diff -u %t/deps.json %t/deps2.json


// RUN: c-index-test core -materialize-cached-job -cas-path %t/cas @%t/cache-key \
// RUN:   -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext \
// RUN:   -fcas-plugin-option upstream-path=%t/cas-upstream -fcas-plugin-option no-logging

// RUN: c-index-test core -replay-cached-job -cas-path %t/cas @%t/cache-key \
// RUN:   -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext \
// RUN:   -fcas-plugin-option no-logging \
// RUN:   -working-dir %t \
// RUN: -- @%t/cc1.rsp \
// RUN:   -serialize-diagnostic-file %t/t2.dia -Rcompile-job-cache-hit \
// RUN:   -dependency-file %t/t2.d -o %t/output2.o 2> %t/output2.txt

// RUN: diff %t/output1.o %t/output2.o
// RUN: diff -u %t/output1.txt %t/output2.txt
// RUN: diff %t/t1.dia %t/t2.dia
// RUN: diff -u %t/t1.d %t/t2.d

// Check with different `-working-dir` flag.
// RUN: mkdir -p %t/a/b
// RUN: cd %t/a
// RUN: c-index-test core -replay-cached-job -cas-path %t/cas @%t/cache-key \
// RUN:   -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext \
// RUN:   -fcas-plugin-option no-logging \
// RUN:   -working-dir %t/a/b \
// RUN: -- @%t/cc1.rsp \
// RUN:   -serialize-diagnostic-file rel.dia -Rcompile-job-cache-hit \
// RUN:   -dependency-file rel.d -o reloutput.o

// RUN: diff %t/output1.o %t/a/b/reloutput.o
// RUN: diff -u %t/t1.d %t/a/b/rel.d
// FIXME: Get clang's `-working-directory` to affect relative path for serialized diagnostics.

// Use relative path to inputs and outputs.
//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -c main.c -target x86_64-apple-macos11 -MD -MF t1.d -MT deps -o output1.o",
  "file": "DIR/main.c"
}]

//--- main.c
#warning some warning
