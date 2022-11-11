# REQUIRES: shell
# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.dir/foo.o %s
# RUN: wasm-ld --reproduce=%t.dir/repro.tar -o %t.dir/out.wasm %t.dir/foo.o
# RUN: env LLD_REPRODUCE=%t.dir/repro2.tar wasm-ld -o %t.dir/out.wasm %t.dir/foo.o

# RUN: cd %t.dir
# RUN: tar tf repro.tar | FileCheck --check-prefix=TAR %s
# RUN: tar tf repro2.tar | FileCheck --check-prefix=TAR2 %s

# TAR: repro/response.txt
# TAR: repro/version.txt
# TAR: repro/{{.*}}/foo.o

# TAR2: repro2/response.txt
# TAR2: repro2/version.txt
# TAR2: repro2/{{.*}}/foo.o

# RUN: tar xf repro.tar
# RUN: FileCheck --check-prefix=RSP %s < repro/response.txt

# RSP: -o {{.*}}out.wasm
# RSP: {{.*}}/foo.o

# RUN: FileCheck %s --check-prefix=VERSION < repro/version.txt
# VERSION: LLD

.globl  _start
_start:
  .functype _start () -> ()
  end_function
