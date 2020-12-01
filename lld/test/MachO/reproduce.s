# REQUIRES: x86

# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir/build1
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t.dir/build1/foo.o
# RUN: cd %t.dir
# RUN: %lld -platform_version macos 10.10.0 11.0 build1/foo.o -o bar --reproduce repro1.tar
# RUN: tar xOf repro1.tar repro1/%:t.dir/build1/foo.o > build1-foo.o
# RUN: cmp build1/foo.o build1-foo.o

# RUN: tar xf repro1.tar repro1/response.txt repro1/version.txt
# RUN: FileCheck %s --check-prefix=RSP1 < repro1/response.txt
# RSP1:      {{^}}-platform_version macos 10.10.0 11.0{{$}}
# RSP1-NOT:  {{^}}repro1{{[/\\]}}
# RSP1-NEXT: {{[/\\]}}foo.o
# RSP1-NEXT: -o bar
# RSP1-NOT:  --reproduce

# RUN: FileCheck %s --check-prefix=VERSION < repro1/version.txt
# VERSION: LLD

# RUN: mkdir -p %t.dir/build2/a/b/c
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t.dir/build2/foo.o
# RUN: cd %t.dir/build2/a/b/c
# RUN: echo ./../../../foo.o > %t.dir/build2/filelist
# RUN: env LLD_REPRODUCE=repro2.tar %lld -filelist %t.dir/build2/filelist -o /dev/null
# RUN: tar xOf repro2.tar repro2/%:t.dir/build2/foo.o > build2-foo.o
# RUN: cmp %t.dir/build2/foo.o build2-foo.o

# RUN: tar xf repro2.tar repro2/response.txt repro2/version.txt
# RUN: FileCheck %s --check-prefix=RSP2 < repro2/response.txt
# RSP2-NOT:  {{^}}repro2{{[/\\]}}
# RSP2:      {{[/\\]}}foo.o

.globl _main
_main:
  ret
