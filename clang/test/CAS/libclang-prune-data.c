// REQUIRES: ondisk_cas

// Tests that the CAS directory storage can be limited via libclang APIs.
// The test depends on internal details of the CAS directory structure.

// RUN: rm -rf %t && mkdir -p %t

// RUN: %clang -cc1depscan -fdepscan=inline -o %t/t.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macos12 -fcas-path %t/cas -emit-obj %s -o %t/output.o
// RUN: %clang @%t/t.rsp
// RUN: ls %t/cas | wc -l | grep 2
// RUN: ls %t/cas | grep v1.1

// Limit too high, no change.
// RUN: c-index-test core -prune-cas -cas-path %t/cas 100000000
// RUN: ls %t/cas | wc -l | grep 2

// Under the limit, starts a chain.
// RUN: c-index-test core -prune-cas -cas-path %t/cas 10
// RUN: ls %t/cas | wc -l | grep 3
// RUN: ls %t/cas | grep v1.2

// Under the limit, starts a chain and abandons oldest dir.
// RUN: c-index-test core -prune-cas -cas-path %t/cas 10
// RUN: ls %t/cas | wc -l | grep 4
// RUN: ls %t/cas | grep v1.3

// Under the limit, removes abandonded dir, starts a chain and abandons oldest dir.
// RUN: c-index-test core -prune-cas -cas-path %t/cas 10
// RUN: ls %t/cas | wc -l | grep 4
// RUN: ls %t/cas | grep v1.4
// RUN: ls %t/cas | grep -v v1.1

// Same test but using the plugin CAS.

// RUN: rm -rf %t/cas

// RUN: %clang -cc1depscan -fdepscan=inline -o %t/t.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macos12 -fcas-path %t/cas -emit-obj %s -o %t/output.o \
// RUN:   -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext
// RUN: %clang @%t/t.rsp
// RUN: ls %t/cas | wc -l | grep 2
// RUN: ls %t/cas | grep v1.1

// Limit too high, no change.
// RUN: c-index-test core -prune-cas -cas-path %t/cas 100000000 -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext
// RUN: ls %t/cas | wc -l | grep 2

// Under the limit, starts a chain.
// RUN: c-index-test core -prune-cas -cas-path %t/cas 10 -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext
// RUN: ls %t/cas | wc -l | grep 3
// RUN: ls %t/cas | grep v1.2

// Under the limit, starts a chain and abandons oldest dir.
// RUN: c-index-test core -prune-cas -cas-path %t/cas 10 -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext
// RUN: ls %t/cas | wc -l | grep 4
// RUN: ls %t/cas | grep v1.3

// Under the limit, removes abandonded dir, starts a chain and abandons oldest dir.
// RUN: c-index-test core -prune-cas -cas-path %t/cas 10 -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext
// RUN: ls %t/cas | wc -l | grep 4
// RUN: ls %t/cas | grep v1.4
// RUN: ls %t/cas | grep -v v1.1
