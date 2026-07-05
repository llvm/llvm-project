# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

## Make sure the option parser doesn't think --x and -w are flags.
# RUN: %lld -dylib -o %t %t.o \
# RUN:      -segprot FOO rwx xwr \
# RUN:      -segprot BAR --x --x \
# RUN:      -segprot BAZ -w -w
# RUN: llvm-readobj --macho-segment %t | FileCheck %s

# CHECK:        Name: FOO
# CHECK-NEXT:   Size:
# CHECK-NEXT:   vmaddr:
# CHECK-NEXT:   vmsize:
# CHECK-NEXT:   fileoff:
# CHECK-NEXT:   filesize:
# CHECK-NEXT:   maxprot: rwx
# CHECK-NEXT:   initprot: rwx

# CHECK:        Name: BAR
# CHECK-NEXT:   Size:
# CHECK-NEXT:   vmaddr:
# CHECK-NEXT:   vmsize:
# CHECK-NEXT:   fileoff:
# CHECK-NEXT:   filesize:
# CHECK-NEXT:   maxprot: --x
# CHECK-NEXT:   initprot: --x

# CHECK:        Name: BAZ
# CHECK-NEXT:   Size:
# CHECK-NEXT:   vmaddr:
# CHECK-NEXT:   vmsize:
# CHECK-NEXT:   fileoff:
# CHECK-NEXT:   filesize:
# CHECK-NEXT:   maxprot: -w-
# CHECK-NEXT:   initprot: -w-

# RUN: %lld -dylib -o %t.different %t.o -segprot FOO rw r
# RUN: llvm-readobj --macho-segment %t.different \
# RUN:     | FileCheck %s --check-prefix=DIFFERENT

# RUN: %no-arg-lld -arch x86_64 -platform_version "mac catalyst" 14.0.0 17.5 \
# RUN:     -dylib -o /dev/null %t.o -segprot FOO rw r
# RUN: llvm-readobj --macho-segment %t.different \
# RUN:     | FileCheck %s --check-prefix=DIFFERENT

# DIFFERENT:        Name: FOO
# DIFFERENT-NEXT:   Size:
# DIFFERENT-NEXT:   vmaddr:
# DIFFERENT-NEXT:   vmsize:
# DIFFERENT-NEXT:   fileoff:
# DIFFERENT-NEXT:   filesize:
# DIFFERENT-NEXT:   maxprot: rw-
# DIFFERENT-NEXT:   initprot: r--

# RUN: not %no-arg-lld -arch x86_64 -platform_version ios-simulator 14.0 15.0 \
# RUN:     -dylib -o /dev/null %t.o -segprot FOO rwx rw 2>&1 \
# RUN:     | FileCheck %s --check-prefix=MISMATCH
# RUN: not %lld -dylib -o /dev/null %t.o -segprot FOO r rw 2>&1 \
# RUN:     | FileCheck %s --check-prefix=INITTOOPERMISSIVE
# RUN: not %lld -dylib -o /dev/null %t.o -segprot __LINKEDIT rwx rwx 2>&1 \
# RUN:     | FileCheck %s --check-prefix=NO-LINKEDIT
# RUN: not %lld -dylib -o /dev/null %t.o -segprot FOO uhh wat 2>&1 \
# RUN:     | FileCheck %s --check-prefix=MISPARSE
# RUN: not %lld -dylib -o /dev/null %t.o -segprot FOO rwx 2>&1 \
# RUN:     | FileCheck %s --check-prefix=MISSING

# MISMATCH:    error: invalid argument '-segprot FOO rwx rw': max and init must be the same for non-macOS non-i386 archs
# INITTOOPERMISSIVE: error: invalid argument '-segprot FOO r rw': init must not be more permissive than max
# NO-LINKEDIT: error: -segprot cannot be used to change __LINKEDIT's protections
# MISPARSE:    error: unknown -segprot letter 'u' in uhh
# MISPARSE:    error: unknown -segprot letter 'a' in wat
# MISSING:     error: -segprot: missing argument

.section FOO,foo
.section BAR,bar
.section BAZ,baz
