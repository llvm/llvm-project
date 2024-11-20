// COM: fail to create the cache, but still produce something valid
// RUN: rm -f %t_log
// RUN: AMD_COMGR_CACHE_DIR=%t.cache \
// RUN:   AMD_COMGR_CACHE_POLICY="foo=2h" \
// RUN:   AMD_COMGR_EMIT_VERBOSE_LOGS=1 \
// RUN:   AMD_COMGR_REDIRECT_LOGS=%t.log \
// RUN:     compile-minimal-test %S/compile-minimal-test.cl %t.bin
// RUN: llvm-objdump -d %t.bin | FileCheck %S/compile-minimal-test.cl 
// RUN: FileCheck --check-prefix=BAD %s < %t.log
// BAD: when parsing the cache policy: Unknown key: 'foo'
//
// COM: the cache has not been created since we couldn't parse the policy
// RUN: [ ! -d %t.cache ]
