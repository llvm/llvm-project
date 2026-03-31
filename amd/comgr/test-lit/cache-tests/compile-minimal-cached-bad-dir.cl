// RUN: export AMD_COMGR_CACHE=1
//
// COM: fail to create the cache, but still produce something valid
// RUN: rm -f %t.log
// RUN: echo "not a directory" >  %t.txt
// RUN: AMD_COMGR_CACHE_DIR=%t.txt \
// RUN:   AMD_COMGR_EMIT_VERBOSE_LOGS=1 \
// RUN:   AMD_COMGR_REDIRECT_LOGS=%t.log \
// RUN:     compile-opencl-minimal %S/../compile-minimal.cl %t.bin 1.2
// RUN: %llvm-objdump -d %t.bin | %FileCheck %S/../compile-minimal.cl
// RUN: %FileCheck --check-prefix=BAD %s < %t.log
// COM: The error message differs by platform:
// COM: Linux:   Comgr cache, when building the add stream callback: Failed to open cache file <path>: Not a directory
// COM: Windows: Comgr cache, when getting the cached file stream: no such file or directory: AMDGPUCompilerCache: Can't get a temporary file
// BAD: Comgr cache,
// BAD-SAME: {{Not a directory|no such file or directory}}
