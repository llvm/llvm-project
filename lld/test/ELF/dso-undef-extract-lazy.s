# REQUIRES: x86
# RUN: rm -fr %t && split-file %s %t

## Build an object with a trivial main function
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/main.s -o %t1.o

## Build %t.a which defines a global 'foo'
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/archive.s -o %t2.o
# RUN: rm -f %t2.a
# RUN: llvm-ar rc %t2.a %t2.o

## Build %t.so that has a reference to 'foo'
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/shlib.s -o %t3.o
# RUN: ld.lld %t3.o -o %t3.so -shared

## Test that 'foo' from %t2.a is fetched to define 'foo' needed by %t3.so.
## Test both cases where the archive is before or after the shared library in
## link order.

# RUN: ld.lld %t1.o %t2.a %t3.so -o %t.exe
# RUN: llvm-readelf --dyn-symbols %t.exe | FileCheck %s --check-prefix=CHECK-FETCH

# RUN: ld.lld %t1.o %t3.so %t2.a -o %t.exe
# RUN: llvm-readelf --dyn-symbols %t.exe | FileCheck %s --check-prefix=CHECK-FETCH

# CHECK-FETCH: GLOBAL DEFAULT {{[0-9]+}} foo

## Unversioned undefined symbols also extract the archive definitions.
# RUN: yaml2obj %t/ver.yaml -o %t4.so
# RUN: ld.lld %t1.o %t4.so %t2.a -o %t.exe
# RUN: llvm-readelf --dyn-symbols %t.exe | FileCheck %s --check-prefix=CHECK-FETCH

#--- main.s
.text
.globl _start
.type _start,@function
_start:
  ret

#--- archive.s
.global foo
foo:

#--- shlib.s
.global foo

#--- ver.yaml
--- !ELF
FileHeader:
  Class: ELFCLASS64
  Data:  ELFDATA2LSB
  Type:  ET_DYN
  Machine: EM_X86_64
Sections:
  - Name:            .gnu.version
    Type:            SHT_GNU_versym
    Flags:           [ SHF_ALLOC ]
    Address:         0x0000000000200210
    AddressAlign:    0x0000000000000002
    EntSize:         0x0000000000000002
## Test both index 0 and 1 for unversioned undefined symbols.
## https://sourceware.org/PR33577
    Entries:         [ 0, 0, 1 ]
  - Name:            .gnu.version_r
    Type:            SHT_GNU_verneed
    Flags:           [ SHF_ALLOC ]
    Address:         0x0000000000200250
    AddressAlign:    0x0000000000000004
    Dependencies:
      - Version:         1
        File:            dso.so.0
        Entries:
          - Name:            v1
            Hash:            1937
            Flags:           0
            Other:           3
DynamicSymbols:
  - Name:    _start
    Binding: STB_GLOBAL
  - Name:    foo
    Binding: STB_GLOBAL
