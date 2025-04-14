// RUN: rm -fr %t.cache
//
// RUN: unset AMD_COMGR_CACHE
// RUN: AMD_COMGR_CACHE_DIR=%t.cache compile-opencl-minimal \
// RUN:    %S/../compile-minimal-test.cl %t.bin 1.2
// RUN: llvm-objdump -d %t.bin | FileCheck %S/../compile-minimal-test.cl
// RUN: [ -d %t.cache ]
//
// RUN: rm -fr %t.cache
//
// RUN: export AMD_COMGR_CACHE=0
// RUN: AMD_COMGR_CACHE_DIR=%t.cache compile-opencl-minimal \
// RUN:    %S/../compile-minimal-test.cl %t.bin 1.2
// RUN: llvm-objdump -d %t.bin | FileCheck %S/../compile-minimal-test.cl
// RUN: [ ! -d %t.cache ]
//
// RUN: export AMD_COMGR_CACHE=1
//
// COM: Run once and check that the cache directory exists and it has more than
// COM     1 element (one for the cache tag, one or more for the cached
// COM:    commands)
// RUN: AMD_COMGR_CACHE_DIR=%t.cache compile-opencl-minimal \
// RUN:    %S/../compile-minimal-test.cl %t_a.bin 1.2
// RUN: llvm-objdump -d %t_a.bin | FileCheck %S/../compile-minimal-test.cl
// RUN: COUNT_BEFORE=$(ls "%t.cache" | wc -l)

// COM: One element for the tag, one for cli->bc, one for bc->obj another
// COM: for obj->exec. No elements for src->cli since this is not supported.
// RUN: [ 4 -eq $COUNT_BEFORE ]
//
// RUN: AMD_COMGR_CACHE_DIR=%t.cache compile-opencl-minimal \
// RUN:    %S/../compile-minimal-test.cl %t_b.bin 1.2
// RUN: llvm-objdump -d %t_b.bin | FileCheck %S/../compile-minimal-test.cl
// RUN: COUNT_AFTER=$(ls "%t.cache" | wc -l)
// RUN: [ $COUNT_AFTER = $COUNT_BEFORE ]
//
