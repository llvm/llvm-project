# REQUIRES: aarch64

## --no-leading-lines is needed for .tbd files.
# RUN: rm -rf %t; split-file --no-leading-lines %s %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o %t/foo.o %t/foo.s
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o %t/bar.o %t/bar.s

## MH_APP_EXTENSION_SAFE is only set on dylibs, and only if requested.
# RUN: %lld -arch arm64 -dylib -o %t/foo.dylib %t/foo.o
# RUN: llvm-otool -hv %t/foo.dylib | FileCheck --check-prefix=NOAPPEXT %s
# RUN: %lld -arch arm64 -dylib -o %t/foo-appext.dylib %t/foo.o \
# RUN:     -application_extension
# RUN: llvm-otool -hv %t/foo-appext.dylib | FileCheck --check-prefix=APPEXT %s
# RUN: %lld -arch arm64 -dylib -o %t/foo-noappext.dylib %t/foo.o \
# RUN:     -application_extension -no_application_extension
# RUN: llvm-otool -hv %t/foo-noappext.dylib \
# RUN:     | FileCheck --check-prefix=NOAPPEXT %s
# RUN: %lld -arch arm64 -bundle -o %t/foo.so %t/foo.o \
# RUN:     -application_extension
# RUN: llvm-otool -hv %t/foo.so | FileCheck --check-prefix=NOAPPEXT %s

# APPEXT: APP_EXTENSION_SAFE
# NOAPPEXT-NOT: APP_EXTENSION_SAFE

## The warning is emitted for all target types.
# RUN: %lld -arch arm64 -dylib -o %t/bar.dylib %t/bar.o \
# RUN:     -application_extension %t/foo-appext.dylib
# RUN: %lld -arch arm64 -dylib -o %t/bar.dylib %t/bar.o \
# RUN:     -application_extension -L %t -ltbd-appext
# RUN: not %lld -arch arm64 -dylib -o %t/bar.dylib %t/bar.o \
# RUN:     -application_extension %t/foo-noappext.dylib \
# RUN:     2>&1 | FileCheck --check-prefix=WARN %s
# RUN: not %lld -arch arm64 -dylib -o %t/bar.dylib %t/bar.o \
# RUN:     -application_extension -L %t -ltbd-noappext \
# RUN:     2>&1 | FileCheck --check-prefix=WARN %s
# RUN: not %lld -arch arm64 -bundle -o %t/bar.so %t/bar.o \
# RUN:     -application_extension %t/foo-noappext.dylib \
# RUN:     2>&1 | FileCheck --check-prefix=WARN %s
# RUN: not %lld -arch arm64 -bundle -o %t/bar.so %t/bar.o \
# RUN:     -application_extension -L %t -ltbd-noappext \
# RUN:     2>&1 | FileCheck --check-prefix=WARN %s

# WARN: using '-application_extension' with unsafe dylib:

## Test we warn on dylibs loaded indirectly via reexports.
# RUN: not %lld -arch arm64 -dylib -o %t/bar.dylib %t/bar.o \
# RUN:     -application_extension -L %t -lbaz-noappext-reexport \
# RUN:     -u _baz 2>&1 | FileCheck --check-prefix=WARN %s

#--- foo.s
.globl _foo
.p2align 2
_foo:
  ret

#--- libtbd-appext.tbd
--- !tapi-tbd
tbd-version: 4
targets: [ arm64-macos ]
uuids:
  - target: arm64-macos
    value:  2E994C7F-3F03-3A07-879C-55690D22BEDA
install-name:     '/usr/lib/libtbd-appext.dylib'
exports:
  - targets:      [ arm64-macos ]
    symbols:      [ _foo ]
...

#--- libtbd-noappext.tbd
--- !tapi-tbd
tbd-version: 4
targets: [ arm64-macos ]
flags: [ not_app_extension_safe ]
uuids:
  - target: arm64-macos
    value:  2E994C7F-3F03-3A07-879C-55690D22BEDA
install-name:     '/usr/lib/libtbd-noappext.dylib'
exports:
  - targets:      [ arm64-macos ]
    symbols:      [ _foo ]
...

#--- bar.s
.globl _bar
.p2align 2
_bar:
  ret

#--- libbaz-noappext-reexport.tbd
--- !tapi-tbd
tbd-version:      4
targets:          [ arm64-macos ]
uuids:
  - target:       arm64-macos
    value:        00000000-0000-0000-0000-000000000001
install-name:     '/usr/lib/libbaz.dylib'
reexported-libraries:
  - targets:      [ arm64-macos ]
    libraries:    [ '/usr/lib/libbaz-noappext-reexported.dylib']
--- !tapi-tbd
tbd-version:      4
targets:          [ arm64-macos ]
flags: [ not_app_extension_safe ]
uuids:
  - target:       arm64-macos
    value:        00000000-0000-0000-0000-000000000003
install-name:     '/usr/lib/libbaz-noappext-reexported.dylib'
parent-umbrella:
  - targets:      [ arm64-macos ]
    umbrella:     baz
exports:
  - targets:      [ arm64-macos ]
    symbols:      [ _baz ]
...
