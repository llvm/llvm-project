// REQUIRES: ondisk_cas

// RUN: rm -rf %t && mkdir -p %t

// RUN: %clang -cc1depscan -o %t/t1.rsp -fdepscan=inline -cc1-args \
// RUN:   -cc1 -emit-obj %s -fcas-path %t/cas \
// RUN:   -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext \
// RUN:   -fcas-plugin-option first-prefix=myfirst- -fcas-plugin-option second-prefix=mysecond- \
// RUN:   -fcas-plugin-option upstream-path=%t/cas-upstream
// RUN: %clang @%t/t1.rsp -o %t/t1.o -Rcompile-job-cache 2>&1 | FileCheck %s --check-prefix=CACHE-MISS

// Clear the CAS and check the outputs can still be "downloaded" from upstream.
// RUN: rm -rf %t/cas
// RUN: %clang -cc1depscan -o %t/t2.rsp -fdepscan=inline -cc1-args \
// RUN:   -cc1 -emit-obj %s -fcas-path %t/cas \
// RUN:   -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext \
// RUN:   -fcas-plugin-option first-prefix=myfirst- -fcas-plugin-option second-prefix=mysecond- \
// RUN:   -fcas-plugin-option upstream-path=%t/cas-upstream
// RUN: %clang @%t/t2.rsp -o %t/t2.o -Rcompile-job-cache 2>&1 | FileCheck %s --check-prefix=CACHE-HIT
// RUN: diff %t/t1.o %t/t2.o

// Check that it's a cache miss if outputs are not found in the upstream CAS.
// RUN: rm -rf %t/cas
// RUN: %clang -cc1depscan -o %t/t3.rsp -fdepscan=inline -cc1-args \
// RUN:   -cc1 -emit-obj %s -fcas-path %t/cas \
// RUN:   -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext \
// RUN:   -fcas-plugin-option first-prefix=myfirst- -fcas-plugin-option second-prefix=mysecond- \
// RUN:   -fcas-plugin-option upstream-path=%t/cas-upstream \
// RUN:   -fcas-plugin-option simulate-missing-objects
// RUN: %clang @%t/t3.rsp -o %t/t.o -Rcompile-job-cache 2>&1 | FileCheck %s --check-prefix=CACHE-NOTFOUND

// CACHE-MISS: remark: compile job cache miss for 'myfirst-mysecond-
// CACHE-MISS: warning: some warning

// Check that outputs are downloaded concurrently.
// CACHE-HIT:      load_object_async downstream begin:
// CACHE-HIT-NEXT: load_object_async downstream begin:
// CACHE-HIT-NEXT: load_object_async downstream end:
// CACHE-HIT-NEXT: load_object_async downstream end:
// CACHE-HIT-NEXT: remark: compile job cache hit for 'myfirst-mysecond-
// CACHE-HIT-NEXT: warning: some warning

// CACHE-NOTFOUND: remark: compile job cache backend did not find output 'main' for key
// CACHE-NOTFOUND: remark: compile job cache miss
// CACHE-NOTFOUND: warning: some warning

// RUN: not %clang -cc1depscan -o %t/t.rsp -fdepscan=inline -cc1-args \
// RUN:   -cc1 %s -fcas-path %t/cas \
// RUN:   -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext \
// RUN:   -fcas-plugin-option no-such-option=2 2>&1 | FileCheck %s --check-prefix=FAIL-PLUGIN-OPT
// FAIL-PLUGIN-OPT: fatal error: CAS cannot be initialized from the specified '-fcas-*' options: unknown option: no-such-option

// RUN: not %clang -cc1depscan -o %t/t.rsp -fdepscan=inline -cc1-args \
// RUN:   -cc1 %s -fcas-path %t/cas -fcas-plugin-path %t/non-existent 2>&1 | FileCheck %s --check-prefix=NOTEXISTENT
// NOTEXISTENT: fatal error: CAS cannot be initialized from the specified '-fcas-*' options

#warning some warning
void test() {}
