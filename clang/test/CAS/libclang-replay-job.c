// REQUIRES: shell

// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-cas --cas %t/cas --ingest --data %s > %t/casid
//
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext \
// RUN:   -fcas-plugin-option upstream-path=%t/cas-upstream -fcas-plugin-option no-logging \
// RUN:   -dependency-file %t/t1.d -MT deps -emit-obj -o %t/output1.o %s
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext \
// RUN:   -fcas-plugin-option upstream-path=%t/cas-upstream -fcas-plugin-option no-logging \
// RUN:   -serialize-diagnostic-file %t/t1.dia -dependency-file %t/t1.d -MT deps \
// RUN:   -Rcompile-job-cache-hit -emit-obj -o %t/output1.o %s 2> %t/output1.txt

// Verify the warning was recorded and we compare populated .dia files.
// RUN: c-index-test -read-diagnostics %t/t1.dia 2>&1 | FileCheck %s --check-prefix=DIAGS
// DIAGS: warning: some warning

// RUN: cat %t/output1.txt | grep llvmcas | sed \
// RUN:   -e "s/^.*hit for '//" \
// RUN:   -e "s/' .*$//" > %t/cache-key

// Delete the "local" cache and use the "upstream" one to re-materialize the outputs locally.
// RUN: rm -rf %t/cas
// RUN: c-index-test core -materialize-cached-job -cas-path %t/cas @%t/cache-key \
// RUN:   -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext \
// RUN:   -fcas-plugin-option upstream-path=%t/cas-upstream -fcas-plugin-option no-logging

// RUN: c-index-test core -replay-cached-job -cas-path %t/cas @%t/cache-key \
// RUN:   -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext \
// RUN:   -fcas-plugin-option no-logging \
// RUN: -- -cc1 \
// RUN:   -serialize-diagnostic-file %t/t2.dia -Rcompile-job-cache-hit \
// RUN:   -dependency-file %t/t2.d -MT deps \
// RUN:   -o %t/output2.o 2> %t/output2.txt

// RUN: diff %t/output1.o %t/output2.o
// RUN: diff -u %t/output1.txt %t/output2.txt
// RUN: diff %t/t1.dia %t/t2.dia
// RUN: diff -u %t/t1.d %t/t2.d

// Check with `-working-dir` flag.
// RUN: mkdir -p %t/a/b
// RUN: cd %t/a
// RUN: c-index-test core -replay-cached-job -cas-path %t/cas @%t/cache-key \
// RUN:   -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext \
// RUN:   -fcas-plugin-option no-logging \
// RUN:   -working-dir %t/a/b \
// RUN: -- -cc1 %t/a/b \
// RUN:   -serialize-diagnostic-file rel.dia -Rcompile-job-cache-hit \
// RUN:   -dependency-file rel.d -MT deps \
// RUN:   -o reloutput.o

// RUN: diff %t/output1.o %t/a/b/reloutput.o
// RUN: diff -u %t/t1.d %t/a/b/rel.d
// FIXME: Get clang's `-working-directory` to affect relative path for serialized diagnostics.

#warning some warning
