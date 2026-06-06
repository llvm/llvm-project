// REQUIRES: x86,aarch64
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %ta.o
// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %s -o %tb.o
// RUN: ld.lld -shared %tb.o -o %ti686.so
// RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux %s -o %tc.o

// RUN: not ld.lld %ta.o %tb.o -o /dev/null 2>&1 | \
// RUN:   FileCheck --check-prefix=A-AND-B %s
// A-AND-B: error: {{.*}}b.o is incompatible with {{.*}}a.o

// RUN: ld.lld --noinhibit-exec %tb.o %tc.o -o /dev/null 2>&1 | \
// RUN:   FileCheck --check-prefix=B-AND-C %s
// B-AND-C: warning: {{.*}}c.o is incompatible with {{.*}}b.o

// RUN: not ld.lld %ta.o %ti686.so -o /dev/null 2>&1 | \
// RUN:   FileCheck --check-prefix=A-AND-SO %s
// A-AND-SO: i686.so is incompatible with {{.*}}a.o

// RUN: not ld.lld %tc.o %ti686.so -o /dev/null 2>&1 | \
// RUN:   FileCheck --check-prefix=C-AND-SO %s
// C-AND-SO: i686.so is incompatible with {{.*}}c.o

// RUN: not ld.lld %ti686.so %tc.o -o /dev/null 2>&1 | \
// RUN:   FileCheck --check-prefix=SO-AND-C %s
// SO-AND-C: c.o is incompatible with {{.*}}i686.so

// RUN: not ld.lld -m elf64ppc %ta.o -o /dev/null 2>&1 | \
// RUN:   FileCheck --check-prefix=A-ONLY %s
// A-ONLY: a.o is incompatible with elf64ppc

// RUN: not ld.lld -m elf64ppc %tb.o -o /dev/null 2>&1 | \
// RUN:   FileCheck --check-prefix=B-ONLY %s
// B-ONLY: b.o is incompatible with elf64ppc

// RUN: not ld.lld -m elf64ppc %tc.o -o /dev/null 2>&1 | \
// RUN:   FileCheck --check-prefix=C-ONLY %s
// C-ONLY: c.o is incompatible with elf64ppc

// RUN: not ld.lld -m elf_i386 %tc.o %ti686.so -o /dev/null 2>&1 | \
// RUN:   FileCheck --check-prefix=C-AND-SO-I386 %s
// C-AND-SO-I386: c.o is incompatible with elf_i386

// RUN: not ld.lld -m elf_i386 %ti686.so %tc.o -o /dev/null 2>&1 | \
// RUN:   FileCheck --check-prefix=SO-AND-C-I386 %s
// SO-AND-C-I386: c.o is incompatible with elf_i386

// RUN: echo 'OUTPUT_FORMAT(elf32-i386)' > %t.script
// RUN: not ld.lld %t.script %ta.o -o /dev/null 2>&1 | \
// RUN:   FileCheck --check-prefix=A-AND-SCRIPT %s
// RUN: not ld.lld %ta.o %t.script -o /dev/null 2>&1 | \
// RUN:   FileCheck --check-prefix=A-AND-SCRIPT %s
// RUN: not ld.lld -m elf_x86_64 %ta.o %t.script -o /dev/null 2>&1 | \
// RUN:   FileCheck --check-prefix=A-AND-SCRIPT %s
// A-AND-SCRIPT: a.o is incompatible with elf32-i386

// RUN: echo 'OUTPUT_FORMAT(elf32-i386-freebsd)' > %t-freebsd.script
// RUN: not ld.lld %t-freebsd.script %ta.o -o /dev/null 2>&1 | \
// RUN:   FileCheck --check-prefix=A-AND-FREEBSD-SCRIPT %s
// A-AND-FREEBSD-SCRIPT: a.o is incompatible with elf32-i386-freebsd

/// %tb.a is not extracted, but we report an error anyway.
// RUN: rm -f %tb.a && llvm-ar rc %tb.a %tb.o
// RUN: not ld.lld %ta.o %tb.a -o /dev/null 2>&1 | FileCheck --check-prefix=UNEXTRACTED-ARCHIVE %s
// UNEXTRACTED-ARCHIVE: {{.*}}.a({{.*}}b.o) is incompatible with {{.*}}a.o

// We used to fail to identify this incompatibility and crash trying to
// read a 64 bit file as a 32 bit one.
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/archive2.s -o %tc.o
// RUN: rm -f %t.a
// RUN: llvm-ar rc %t.a %tc.o
// RUN: llvm-mc -filetype=obj -triple=i686-linux %s -o %td.o
// RUN: ld.lld --noinhibit-exec %t.a %td.o 2>&1 -o /dev/null | FileCheck --check-prefix=ARCHIVE %s
// ARCHIVE: warning: {{.*}}d.o is incompatible{{$}}
.global _start
_start:
.data
        .long foo

