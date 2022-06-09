# REQUIRES: x86

# RUN: rm -rf %t; split-file --no-leading-lines %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/ref_xxx.s -o %t/ref_xxx.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/ref_ySyy.s -o %t/ref_ySyy.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/ref_zzz.s -o %t/ref_zzz.o

## Case 1: special symbol $ld$previous affects the install name / compatibility version
## since the specified version 11.0.0 is within the affected range [3.0, 14.0).

# RUN: %lld -o %t/libfoo1.dylib %t/libLDPreviousInstallName.tbd %t/ref_xxx.o -dylib -platform_version macos 11.0.0 11.0.0
# RUN: llvm-objdump --macho --dylibs-used %t/libfoo1.dylib | FileCheck --check-prefix=CASE1 %s
# CASE1: /Old (compatibility version 1.2.3, current version 5.0.0)

## Case 2: special symbol $ld$previous does not affect the install name / compatibility version
## since the specified version 2.0.0 is lower than the affected range [3.0, 14.0).

# RUN: %lld -o %t/libfoo2.dylib %t/libLDPreviousInstallName.tbd %t/ref_xxx.o -dylib -platform_version macos 2.0.0 2.0.0
# RUN: llvm-objdump --macho --dylibs-used %t/libfoo2.dylib | FileCheck --check-prefix=CASE2 %s
# CASE2: /New (compatibility version 1.1.1, current version 5.0.0)

## Case 3: special symbol $ld$previous does not affect the install name / compatibility version
## since the specified version 14.0.0 is higher than the affected range [3.0, 14.0).

# RUN: %lld -o %t/libfoo3.dylib %t/libLDPreviousInstallName.tbd %t/ref_xxx.o -dylib -platform_version macos 2.0.0 2.0.0
# RUN: llvm-objdump --macho --dylibs-used %t/libfoo3.dylib | FileCheck --check-prefix=CASE3 %s
# CASE3: /New (compatibility version 1.1.1, current version 5.0.0)

## The remaining cases test handling when a symbol name is part of $ld$previous.

## Case 4: special symbol $ld$previous affects the install name / compatibility version
## when the specified version 11.0.0 is within the affected range [3.0, 14.0) when a symbol
## is part of $previous$ if and only if that named symbol is referenced.
## That is, for $ld$previous$/NewName$$3.0$14.0$_symNam$, if _symNam is
## referenced, it refers to dylib /NewName if the deployment target is
## in [3.0, 14.0).

# RUN: %lld -o %t/libfoo4_yes.dylib %t/libLDPreviousInstallName-Symbol.tbd %t/ref_ySyy.o -dylib -platform_version macos 11.0.0 11.0.0
# RUN: llvm-otool -L %t/libfoo4_yes.dylib | FileCheck --check-prefix=CASE4-YES --implicit-check-not=/New %s
# CASE4-YES: /Old (compatibility version 1.2.3, current version 1.2.3)

## $previous has no effect because deployment target is too new.
# RUN: %lld -o %t/libfoo4_no.dylib %t/libLDPreviousInstallName-Symbol.tbd %t/ref_ySyy.o -dylib -platform_version macos 14.0.0 14.0.0
# RUN: llvm-otool -L %t/libfoo4_no.dylib | FileCheck --check-prefix=CASE4-NO --implicit-check-not=/Old %s
# CASE4-NO: /New (compatibility version 1.1.1, current version 5.0.0)

## $previous has no effect because named symbol isn't referenced.
# RUN: %lld -o %t/libfoo4_no.dylib %t/libLDPreviousInstallName-Symbol.tbd %t/ref_zzz.o -dylib -platform_version macos 11.0.0 11.0.0
# RUN: llvm-otool -L %t/libfoo4_no.dylib | FileCheck --check-prefix=CASE4-NO %s

## Case 5: Reference two symbols that add different $previous names each,
## and one that references the "normal" dylib.
## This should produce three different load commands.
# RUN: %lld -o %t/libfoo5.dylib %t/libLDPreviousInstallName-Symbol.tbd %t/ref_xxx.o %t/ref_ySyy.o %t/ref_zzz.o -dylib -platform_version macos 11.0.0 11.0.0
# RUN: llvm-otool -L %t/libfoo5.dylib | FileCheck --check-prefix=CASE5 %s
# CASE5: /New (compatibility version 1.1.1, current version 5.0.0)
# CASE5-DAG: /Another (compatibility version 1.1.1, current version 5.0.0)
# CASE5-DAG: /Old (compatibility version 1.2.3, current version 1.2.3)

## Check that we emit a warning for an invalid start, end and compatibility versions.

# RUN: %no-fatal-warnings-lld -o %t/libfoo1.dylib %t/libLDPreviousInvalid.tbd %t/ref_xxx.o -dylib \
# RUN:  -platform_version macos 11.0.0 11.0.0 2>&1 | FileCheck --check-prefix=INVALID-VERSION %s

# INVALID-VERSION-DAG: failed to parse start version, symbol '$ld$previous$/New$1.2.3$1$3.a$14.0$$' ignored
# INVALID-VERSION-DAG: failed to parse end version, symbol '$ld$previous$/New$1.2.3$1$3.0$14.b$$' ignored
# INVALID-VERSION-DAG: failed to parse compatibility version, symbol '$ld$previous$/New$1.2.c$1$3.0$14.0$$' ignored

#--- ref_xxx.s
.long	_xxx@GOTPCREL

#--- ref_ySyy.s
.long	_y$yy@GOTPCREL

#--- ref_zzz.s
.long	_zzz@GOTPCREL

#--- libLDPreviousInstallName.tbd
--- !tapi-tbd-v3
archs:           [ x86_64 ]
uuids:           [ 'x86_64: 19311019-01AB-342E-812B-73A74271A715' ]
platform:        macosx
install-name:    '/New'
current-version: 5
compatibility-version: 1.1.1
exports:
  - archs:           [ x86_64 ]
    symbols:         [ '$ld$previous$/Old$1.2.3$1$3.0$14.0$$', _xxx ]
...

#--- libLDPreviousInstallName-Symbol.tbd
--- !tapi-tbd-v3
archs:           [ x86_64 ]
uuids:           [ 'x86_64: 19311019-01AB-342E-812B-73A74271A715' ]
platform:        macosx
install-name:    '/New'
current-version: 5
compatibility-version: 1.1.1
exports:
  - archs:           [ x86_64 ]
    symbols:         [
      '$ld$previous$/Another$$1$3.0$14.0$_xxx$',
      '$ld$previous$/Old$1.2.3$1$3.0$14.0$_y$yy$',
      _xxx,
      '_y$yy',
      _zzz,
    ]
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
