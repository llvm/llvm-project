// REQUIRES: x86

// Check that we can wrap a dllimported symbol, so that references to
// __imp_<symbol> gets redirected to a defined local import instead.

// RUN: split-file %s %t.dir
// RUN: llvm-mc -filetype=obj -triple=i686-win32-gnu %t.dir/main.s -o %t.main.obj
// RUN: llvm-mc -filetype=obj -triple=i686-win32-gnu %t.dir/other.s -o %t.other.obj

// RUN: lld-link -dll -out:%t.dll %t.other.obj -noentry -safeseh:no -export:foo -implib:%t.lib
// RUN: lld-link -out:%t.exe %t.main.obj %t.lib -entry:entry -subsystem:console -debug:symtab -safeseh:no -wrap:foo -lldmap:%t.map
// RUN: llvm-objdump -s -d --print-imm-hex %t.exe | FileCheck %s

// CHECK:      Contents of section .rdata:
// CHECK-NEXT:  402000 06104000

// CHECK:      Disassembly of section .text:
// CHECK-EMPTY:
// CHECK:      00401000 <_entry>:
// CHECK-NEXT:   401000: ff 25 00 20 40 00             jmpl    *0x402000
// CHECK-EMPTY:
// CHECK-NEXT: 00401006 <___wrap_foo>:
// CHECK-NEXT:   401006: c3                            retl

// The jmpl instruction in _entry points at an address in 0x402000,
// which is the first 4 bytes of the .rdata section (above), which is a
// pointer that points at ___wrap_foo.

#--- main.s
.global _entry
_entry:
  jmpl *__imp__foo

.global ___wrap_foo
___wrap_foo:
  ret

#--- other.s
.global _foo

_foo:
  ret
