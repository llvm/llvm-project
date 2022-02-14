# REQUIRES: x86

# RUN: rm -rf %t; split-file --no-leading-lines %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o

## Case 1: special symbol $ld$previous affects the install name / compatibility version
## since the specified version 11.0.0 is within the affected range [3.0, 14.0).

# RUN: %lld -o %t/libfoo1.dylib %t/libLDPreviousInstallName.tbd %t/foo.o -dylib -platform_version macos 11.0.0 11.0.0
# RUN: llvm-objdump --macho --dylibs-used %t/libfoo1.dylib | FileCheck --check-prefix=CASE1 %s
# CASE1: /New (compatibility version 1.2.3, current version 5.0.0)

## Case 2: special symbol $ld$previous does not affect the install name / compatibility version
## since the specified version 2.0.0 is lower than the affected range [3.0, 14.0).

# RUN: %lld -o %t/libfoo2.dylib %t/libLDPreviousInstallName.tbd %t/foo.o -dylib -platform_version macos 2.0.0 2.0.0
# RUN: llvm-objdump --macho --dylibs-used %t/libfoo2.dylib | FileCheck --check-prefix=CASE2 %s
# CASE2: /Old (compatibility version 1.1.1, current version 5.0.0)

## Case 3: special symbol $ld$previous does not affect the install name / compatibility version
## since the specified version 14.0.0 is higher than the affected range [3.0, 14.0).

# RUN: %lld -o %t/libfoo3.dylib %t/libLDPreviousInstallName.tbd %t/foo.o -dylib -platform_version macos 2.0.0 2.0.0
# RUN: llvm-objdump --macho --dylibs-used %t/libfoo3.dylib | FileCheck --check-prefix=CASE3 %s
# CASE3: /Old (compatibility version 1.1.1, current version 5.0.0)

## Check that we emit a warning for an invalid start, end and compatibility versions.

# RUN: %no-fatal-warnings-lld -o %t/libfoo1.dylib %t/libLDPreviousInvalid.tbd %t/foo.o -dylib \
# RUN:  -platform_version macos 11.0.0 11.0.0 2>&1 | FileCheck --check-prefix=INVALID-VERSION %s

# INVALID-VERSION-DAG: failed to parse start version, symbol '$ld$previous$/New$1.2.3$1$3.a$14.0$$' ignored
# INVALID-VERSION-DAG: failed to parse end version, symbol '$ld$previous$/New$1.2.3$1$3.0$14.b$$' ignored
# INVALID-VERSION-DAG: failed to parse compatibility version, symbol '$ld$previous$/New$1.2.c$1$3.0$14.0$$' ignored

#--- foo.s
.long	_xxx@GOTPCREL

#--- libLDPreviousInstallName.tbd
--- !tapi-tbd-v3
archs:           [ x86_64 ]
uuids:           [ 'x86_64: 19311019-01AB-342E-812B-73A74271A715' ]
platform:        macosx
install-name:    '/Old'
current-version: 5
compatibility-version: 1.1.1
exports:
  - archs:           [ x86_64 ]
    symbols:         [ '$ld$previous$/New$1.2.3$1$3.0$14.0$$', _xxx ]
...

#--- libLDPreviousInvalid.tbd
--- !tapi-tbd-v3
archs:           [ x86_64 ]
uuids:           [ 'x86_64: 19311019-01AB-342E-112B-73A74271A715' ]
platform:        macosx
install-name:    '/Old'
current-version: 5
compatibility-version: 1.1.1
exports:
  - archs:           [ x86_64 ]
    symbols:         [ '$ld$previous$/New$1.2.3$1$3.a$14.0$$',
                       '$ld$previous$/New$1.2.3$1$3.0$14.b$$',
                       '$ld$previous$/New$1.2.c$1$3.0$14.0$$',
                       _xxx ]
...
