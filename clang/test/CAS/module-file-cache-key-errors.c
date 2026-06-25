// Exercises error paths when -fmodule-file-cache-key resolves to a malformed
// compile-job result object.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t && mkdir -p %t
// RUN: touch %t/tu.c

// RUN: echo -n "llvm::cas::schema::compile_job_result::v1" > %t/schema-name
// RUN: llvm-cas --cas %t/cas --make-blob --data %t/schema-name > %t/schema-id

// Build a "result" that has the right KindRef but no outputs.
// RUN: touch %t/empty
// RUN: llvm-cas --cas %t/cas --make-node --data %t/empty @%t/schema-id > %t/result-empty-id

// Build an invalid result (no KindRef).
// RUN: echo "not-a-result" > %t/garbage
// RUN: llvm-cas --cas %t/cas --make-blob --data %t/garbage > %t/result-garbage-id

// Create cache keys pointing to the results.
// RUN: echo "k1" > %t/key1-data
// RUN: llvm-cas --cas %t/cas --make-blob --data %t/key1-data > %t/key1-id
// RUN: echo "k2" > %t/key2-data
// RUN: llvm-cas --cas %t/cas --make-blob --data %t/key2-data > %t/key2-id
// RUN: llvm-cas --cas %t/cas --put-cache-key @%t/key1-id @%t/result-empty-id
// RUN: llvm-cas --cas %t/cas --put-cache-key @%t/key2-id @%t/result-garbage-id

// RUN: not %clang_cc1 -fcas-path %t/cas -fsyntax-only -fmodule-file-cache-key fake.pcm @%t/key1-id %t/tu.c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EMPTY
// EMPTY: error: module file 'fake.pcm' not found: unloadable module cache key llvmcas://{{[[:xdigit:]]+}}: cached module missing main output

// RUN: not %clang_cc1 -fcas-path %t/cas -fsyntax-only -fmodule-file-cache-key fake.pcm @%t/key2-id %t/tu.c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BOGUS
// BOGUS: error: module file 'fake.pcm' not found: unloadable module cache key llvmcas://{{[[:xdigit:]]+}}: not a compile job result
