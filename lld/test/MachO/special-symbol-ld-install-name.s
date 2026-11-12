# REQUIRES: x86

# RUN: rm -rf %t; split-file --no-leading-lines %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o

## Case 1: special symbol $ld$install_name affects the install name
## since the specified version 11.0.0 matches the target version 11.0.0

# RUN: %lld -o %t/libfoo1.dylib %t/libLDInstallName.tbd %t/foo.o -dylib -platform_version macos 11.0.0 11.0.0
# RUN: llvm-otool -L %t/libfoo1.dylib | FileCheck --check-prefix=CASE1 %s
# CASE1: /New (compatibility version 1.1.1, current version 5.0.0)

## Case 2: special symbol $ld$install_name does not affect the install name
## since the specified version 11.0.0 does not match the target version 12.0.0

# RUN: %lld -o %t/libfoo2.dylib %t/libLDInstallName.tbd %t/foo.o -dylib -platform_version macos 12.0.0 12.0.0
# RUN: llvm-otool -L %t/libfoo2.dylib | FileCheck --check-prefix=CASE2 %s
# CASE2: /Old (compatibility version 1.1.1, current version 5.0.0)

## Check that we emit a warning for an invalid os version.

# RUN: %no-fatal-warnings-lld -o %t/libfoo3.dylib %t/libLDInstallNameInvalid.tbd %t/foo.o -dylib \
# RUN:  -platform_version macos 11.0.0 11.0.0 2>&1 | FileCheck --check-prefix=INVALID-VERSION %s

# INVALID-VERSION: libLDInstallNameInvalid.tbd(/Old): failed to parse os version, symbol '$ld$install_name$os11.a$/New' ignored

## Case 3: If there's another library that has '/New' as its original
## install_name, we should take current-version and compatibility-version from
## the first library we load with that install name. For a reexport,
## the first loaded library is the library doing the reexport, not the
## reexported library.
# RUN: %lld -o %t/libfoo3.dylib %t/libLDInstallNameNew.tbd %t/foo.o -dylib -platform_version macos 11.0.0 11.0.0
# RUN: llvm-otool -L %t/libfoo3.dylib | FileCheck --check-prefix=CASE3 %s
# RUN: %lld -o %t/libfoo3.dylib %t/libLDInstallName.tbd %t/libLDInstallNameNew.tbd %t/foo.o -dylib -platform_version macos 11.0.0 11.0.0
# RUN: llvm-otool -L %t/libfoo3.dylib | FileCheck --check-prefix=CASE1 %s
# RUN: %lld -o %t/libfoo3.dylib %t/libLDInstallNameNew.tbd %t/libLDInstallName.tbd %t/foo.o -dylib -platform_version macos 11.0.0 11.0.0
# RUN: llvm-otool -L %t/libfoo3.dylib | FileCheck --check-prefix=CASE3 %s
# CASE3: /New (compatibility version 4.5.6, current version 9.0.0)

#--- foo.s
.long	_xxx@GOTPCREL

#--- libLDInstallName.tbd
--- !tapi-tbd-v3
archs:           [ x86_64 ]
uuids:           [ 'x86_64: 19311012-01AB-342E-812B-73A74271A715' ]
platform:        macosx
install-name:    '/Old'
current-version: 5
compatibility-version: 1.1.1
exports:
  - archs:           [ x86_64 ]
    symbols:         [ '$ld$install_name$os11.0$/New', _xxx ]
...

#--- libLDInstallNameInvalid.tbd
--- !tapi-tbd-v3
archs:           [ x86_64 ]
uuids:           [ 'x86_64: 19311011-01AB-342E-112B-73A74271A715' ]
platform:        macosx
install-name:    '/Old'
current-version: 5
compatibility-version: 1.1.1
exports:
  - archs:           [ x86_64 ]
    symbols:         [ '$ld$install_name$os11.a$/New', _xxx ]
...

#--- libLDInstallNameNew.tbd
--- !tapi-tbd
tbd-version: 4
targets: [ x86_64-macos ]
uuids:
  - target: x86_64-macos
    value:  2E994C7F-3F03-3A07-879C-55690D22BEDA
install-name: '/New'
current-version: 9
compatibility-version: 4.5.6
reexported-libraries:
  - targets:         [ x86_64-macos ]
    libraries:       [ '@loader_path/libLDInstallName' ]
...
