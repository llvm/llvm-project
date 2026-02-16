# REQUIRES: x86
# Unsupported on Windows due to maximum path length limitations.
# UNSUPPORTED: system-windows

# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir/build1
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t.dir/build1/foo.o
# RUN: echo '_main' > %t.dir/main.exports
# RUN: echo '_main' > %t.dir/main.order
# RUN: echo 'not a virus' > %t.dir/sectdata.txt
# RUN: cd %t.dir
# RUN: %lld -platform_version macos 10.10.0 11.0 \
# RUN:     -exported_symbols_list main.exports \
# RUN:     -order_file main.order \
# RUN:     -sectcreate __COMPLETELY __legit sectdata.txt \
# RUN:     -rpath /usr/lib/swift \
# RUN:     build1/foo.o -o bar --reproduce repro1.tar

# RUN: tar tf repro1.tar | FileCheck -DPATH='%:t.dir' --check-prefix=LIST %s
# LIST: repro1/response.txt
# LIST: [[PATH]]/main.exports
# LIST: [[PATH]]/build1/foo.o
# LIST: [[PATH]]/main.order
# LIST: [[PATH]]/sectdata.txt

# RUN: tar xf repro1.tar
# RUN: cmp build1/foo.o repro1/%:t.dir/build1/foo.o
# RUN: diff main.exports repro1/%:t.dir/main.exports
# RUN: diff main.order repro1/%:t.dir/main.order
# RUN: diff sectdata.txt repro1/%:t.dir/sectdata.txt
# RUN: FileCheck %s --check-prefix=RSP1 < repro1/response.txt
# RSP1:      {{^}}-platform_version macos 10.10.0 11.0{{$}}
# RSP1-NEXT: -exported_symbols_list [[BASEDIR:.+]]/main.exports
# RSP1-NEXT: -order_file [[BASEDIR]]/main.order
# RSP1-NEXT: -sectcreate __COMPLETELY __legit [[BASEDIR]]/sectdata.txt
# RSP1-NEXT: -rpath /usr/lib/swift
# RSP1-NOT:  {{^}}repro1{{[/\\]}}
# RSP1-NEXT: [[BASEDIR]]/build1/foo.o
# RSP1-NEXT: -o bar
# RSP1-NOT:  --reproduce

# RUN: FileCheck %s --check-prefix=VERSION < repro1/version.txt
# VERSION: LLD

# RUN: cd repro1; %no-arg-lld @response.txt

# RUN: mkdir -p %t.dir/build2/a/b/c
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t.dir/build2/foo.o
# RUN: cd %t.dir/build2/a/b/c
# RUN: echo ./../../../foo.o > %t.dir/build2/filelist
# RUN: env LLD_REPRODUCE=repro2.tar %lld -filelist %t.dir/build2/filelist -o /dev/null
# RUN: tar xf repro2.tar
# RUN: cmp %t.dir/build2/foo.o repro2/%:t.dir/build2/foo.o
# RUN: FileCheck %s --check-prefix=RSP2 < repro2/response.txt
# RSP2-NOT:  {{^}}repro2{{[/\\]}}
# RSP2:      {{[/\\]}}foo.o

# RUN: cd repro2; %no-arg-lld @response.txt

.globl _main
_main:
  ret
