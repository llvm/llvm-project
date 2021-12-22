# REQUIRES: x86

# RUN: rm -rf %t; split-file --no-leading-lines %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/ref-all.s -o %t/ref-all.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/ref-reexported.s -o %t/ref-reexported.o

## Check that the hidden symbols(foo11) can't be referenced from HideFoo.tbd when current version is 11.
# RUN: not %lld -o /dev/null %t/libHideFoo.tbd %t/ref-all.o -dylib -platform_version macos 11.0.0 11.0.0 2>&1 | FileCheck %s --check-prefix=ERROR

## Check that the hidden symbol(foo11) can be referenced when the current version is NOT 11.
# RUN: %lld -o %t/ref-foo-12.dylib %t/libHideFoo.tbd %t/ref-all.o -dylib -platform_version macos 12.0.0 12.0.0
# RUN: llvm-objdump --macho --bind %t/ref-foo-12.dylib | FileCheck %s --check-prefix=HAS-FOO

## Check that when we link multiple tbd files, foo11 comes from the tbd where it is visible.
# RUN: %lld -o %t/ref-all.dylib %t/libHideFoo.tbd %t/libHasFoo.tbd %t/ref-all.o -dylib -platform_version macos 11.0.0 11.0.0
# RUN: llvm-objdump --macho --bind %t/ref-all.dylib | FileCheck %s --check-prefix=FOO

## Check that '$hide$' has no effect on reexported symbols.
# RUN: %lld -o %t/reexport.dylib %t/libReexportSystem2.tbd %t/ref-reexported.o -dylib -platform_version macos 11.0.0 11.0.0
# RUN: llvm-objdump --macho --bind %t/reexport.dylib | FileCheck %s --check-prefix=REEXP

# ERROR:  error: undefined symbol: _OBJC_CLASS_$_foo11

# HAS-FOO: __DATA __data              {{.*}} pointer         0 /HideFoo         _OBJC_CLASS_$_foo11

# FOO:      segment  section            address    type       addend dylib            symbol
# FOO-DAG: __DATA   __data               {{.*}} pointer         0 /HideFoo         _OBJC_CLASS_$_bar
# FOO-DAG: __DATA   __data               {{.*}} pointer         0 /HideFoo         _OBJC_CLASS_$_foo10
# FOO-DAG: __DATA   __data               {{.*}} pointer         0 /HasFoo          _OBJC_CLASS_$_foo11
# FOO-DAG: __DATA   __data               {{.*}} pointer         0 /HideFoo         _xxx

# REEXP: __DATA   __data             {{.*}} pointer         0 libSystem        ___nan

#--- ref-all.s
.data        
.quad	_xxx
.quad _OBJC_CLASS_$_foo11
.quad _OBJC_CLASS_$_foo10
.quad _OBJC_CLASS_$_bar

#--- ref-reexported.s
.data
.quad   ___nan

#--- libHideFoo.tbd
--- !tapi-tbd
tbd-version: 4
targets: [ x86_64-macos ]
uuids:
  - target: x86_64-macos
    value:  2E994C7F-3F03-3A07-879C-55690D22BEDA
install-name: '/HideFoo'
current-version: 9
compatibility-version: 4.5.6
exports:
  - targets:         [ x86_64-macos ]
    symbols: [ '$ld$hide$os11.0$_OBJC_CLASS_$_foo11', '$ld$hide$os10.0$_OBJC_CLASS_$_foo10',  _xxx ]
    objc-classes: [foo10, foo11, bar]
...

#--- libHasFoo.tbd
--- !tapi-tbd
tbd-version: 4
targets: [ x86_64-macos ]
uuids:
  - target: x86_64-macos
    value:  2E994C7F-3F03-3A07-879C-55690D22BEDB
install-name: '/HasFoo'
current-version: 9
compatibility-version: 4.5.6
exports:
  - targets: [ x86_64-macos ]
    symbols: [  _xxx ]
    objc-classes: [foo10, foo11, bar]
...

#--- libReexportSystem2.tbd
--- !tapi-tbd
tbd-version:     4
targets: [ x86_64-macos ]
uuids:
  - target:  x86_64-macos
    value:   00000000-0000-0000-0000-000000000002
install-name:    '/libReexportSystem2'
current-version: 9
exports:
  - targets: [ x86_64-macos ]
    symbols: [  '$ld$hide$___nan' ]
reexported-libraries:
  - targets:   [ x86_64-macos ]
    libraries: [ '/usr/lib/libSystem.dylib' ]
...

