# REQUIRES: x86

# RUN: rm -rf %t && mkdir %t

# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %s -o %t/main.o

# RUN: %lld -lSystem -dyld_env DYLD_FRAMEWORK_PATH=./Foo.framework %t/main.o -o %t/one_dyld_env.out
# RUN: llvm-otool -l %t/one_dyld_env.out | FileCheck %s --check-prefix=ONE-ENV

# RUN: %lld -lSystem -dyld_env DYLD_FRAMEWORK_PATH=./Foo.framework \
# RUN:               -dyld_env DYLD_FRAMEWORK_PATH=./Bar.framework \
# RUN:               %t/main.o -o %t/two_dyld_envs.out
# RUN: llvm-otool -l %t/two_dyld_envs.out | FileCheck %s --check-prefix=TWO-ENV

# RUN: not %lld -lSystem -dyld_env DYLD_FRAMEWORK_PATH,./Foo %t/main.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=MALFORMED

# RUN: not %lld  -dylib -lSystem -dyld_env DYLD_FRAMEWORK_PATH=./Foo %t/main.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=DYLIB

# ONE-ENV:     Load command 11
# ONE-ENV-NEXT:          cmd LC_DYLD_ENVIRONMENT
# ONE-ENV-NEXT:      cmdsize 48
# ONE-ENV-NEXT:         name DYLD_FRAMEWORK_PATH=./Foo.framework (offset 12)

# TWO-ENV:      Load command 11
# TWO-ENV-NEXT:          cmd LC_DYLD_ENVIRONMENT
# TWO-ENV-NEXT:      cmdsize 48
# TWO-ENV-NEXT:         name DYLD_FRAMEWORK_PATH=./Foo.framework (offset 12)
# TWO-ENV-NEXT: Load command 12
# TWO-ENV-NEXT:          cmd LC_DYLD_ENVIRONMENT
# TWO-ENV-NEXT:      cmdsize 48
# TWO-ENV-NEXT:         name DYLD_FRAMEWORK_PATH=./Bar.framework (offset 12)

# MALFORMED: error: -dyld_env's argument is malformed. Expected -dyld_env <ENV_VAR>=<VALUE>, got `DYLD_FRAMEWORK_PATH,./Foo`

# DYLIB: error: -dyld_env can only be used when creating executable output

.section __TEXT,__text

.global _main
_main:
  ret
