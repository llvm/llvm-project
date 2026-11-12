# REQUIRES: x86, aarch64
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/x86-64-test.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t/arm64-test.o
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos %s -o %t/arm64-32-test.o
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos %s -o %t/arm64-32-test.o

# RUN: %lld -lSystem -arch x86_64 -o %t/x86-64-executable %t/x86-64-test.o
# RUN: %lld -lSystem -arch arm64 -o %t/arm64-executable %t/arm64-test.o
# RUN: %lld-watchos -lSystem -o %t/arm64-32-executable %t/arm64-32-test.o

# RUN: %lld -arch x86_64 -dylib -o %t/x86-64-dylib %t/x86-64-test.o

## NOTE: recent versions of ld64 don't emit LIB64 for x86-64-executable, maybe we should follow suit
# RUN: llvm-objdump --macho --private-header %t/x86-64-executable | FileCheck %s --check-prefix=EXEC -DCPU=X86_64 -DSUBTYPE=ALL -DCAPS=LIB64
# RUN: llvm-objdump --macho --private-header %t/arm64-executable | FileCheck %s --check-prefix=EXEC -DCPU=ARM64 -DSUBTYPE=ALL -DCAPS=0x00
# RUN: llvm-objdump --macho --private-header %t/arm64-32-executable | FileCheck %s --check-prefix=EXEC -DCPU=ARM64_32 -DSUBTYPE=V8 -DCAPS=0x00

# RUN: llvm-objdump --macho --private-header %t/x86-64-dylib | FileCheck %s --check-prefix=DYLIB -DCPU=X86_64 -DSUBTYPE=ALL -DCAPS=0x00

# EXEC:      magic               cputype  cpusubtype   caps     filetype {{.*}} flags
# EXEC-NEXT: MH_MAGIC{{(_64)?}}  [[CPU]]  [[SUBTYPE]]  [[CAPS]] EXECUTE  {{.*}} NOUNDEFS DYLDLINK TWOLEVEL PIE{{$}}

# DYLIB:      magic                  cputype  cpusubtype   caps      filetype {{.*}} flags
# DYLIB-NEXT: MH_MAGIC_64{{(_64)?}}  [[CPU]]  [[SUBTYPE]]  [[CAPS]]  DYLIB    {{.*}} NOUNDEFS DYLDLINK TWOLEVEL NO_REEXPORTED_DYLIBS{{$}}

.globl _main
_main:
