# REQUIRES: webassembly-registered-target

# RUN: %clang -### %s -c -o tmp.o -target wasm32-unknown-unknown -Wa,--no-type-check 2>&1 | FileCheck %s
# CHECK: "-cc1as" {{.*}} "-mno-type-check"

# Verify that without -Wa,--no-type-check the assembler will error out
# RUN: not %clang %s -c -o tmp.o -target wasm32-unknown-unknown 2>&1 | FileCheck --check-prefix=ERROR %s
# ERROR: error: popped i64, expected i32

foo:
  .functype  foo () -> (i32)
  i64.const 42
  end_function
