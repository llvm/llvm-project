// REQUIRES: ondisk_cas

// RUN: rm -rf %t && mkdir -p %t

// RUN: %clang -cc1depscan -o %t/t.rsp -fdepscan=inline -cc1-args \
// RUN:   -cc1 %s -fcas-path %t/cas \
// RUN:   -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext \
// RUN:   -fcas-plugin-option first-prefix=myfirst- -fcas-plugin-option second-prefix=mysecond-
// RUN: %clang @%t/t.rsp -emit-obj -o %t/t1.o -Rcompile-job-cache 2>&1 | FileCheck %s --check-prefix=CACHE-MISS
// RUN: %clang @%t/t.rsp -emit-obj -o %t/t2.o -Rcompile-job-cache 2>&1 | FileCheck %s --check-prefix=CACHE-HIT
// RUN: diff %t/t1.o %t/t2.o

// CACHE-MISS: remark: compile job cache miss for 'myfirst-mysecond-
// CACHE-MISS: warning: some warning

// CACHE-HIT: remark: compile job cache hit for 'myfirst-mysecond-
// CACHE-HIT: warning: some warning

// RUN: not %clang -cc1depscan -o %t/t.rsp -fdepscan=inline -cc1-args \
// RUN:   -cc1 %s -fcas-path %t/cas \
// RUN:   -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext \
// RUN:   -fcas-plugin-option no-such-option=2 2>&1 | FileCheck %s --check-prefix=FAIL-PLUGIN-OPT
// FAIL-PLUGIN-OPT: fatal error: plugin CAS cannot be initialized {{.*}}: unknown option: no-such-option

// RUN: not %clang -cc1depscan -o %t/t.rsp -fdepscan=inline -cc1-args \
// RUN:   -cc1 %s -fcas-path %t/cas -fcas-plugin-path %t/non-existent 2>&1 | FileCheck %s --check-prefix=NOTEXISTENT
// NOTEXISTENT: fatal error: plugin CAS cannot be initialized

#warning some warning
void test() {}
