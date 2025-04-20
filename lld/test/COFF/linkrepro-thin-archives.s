# REQUIRES: x86

# RUN: rm -rf %t.dir; split-file %s %t.dir

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-windows %t.dir/foo.s -o %t.dir/foo.obj
# RUN: cd %t.dir
# RUN: llvm-ar rcsT foo.lib foo.obj

# RUN: lld-link foo.lib /out:/dev/null /reproduce:repro.tar \
# RUN:     /subsystem:console /machine:x64
# RUN: tar tf repro.tar | FileCheck -DPATH='repro/%:t.dir' %s

# RUN: lld-link /wholearchive foo.lib /out:/dev/null /reproduce:repro2.tar \
# RUN:     /subsystem:console /machine:x64
# RUN: tar tf repro2.tar | FileCheck -DPATH='repro2/%:t.dir' %s

# CHECK-DAG: [[PATH]]/foo.lib
# CHECK-DAG: [[PATH]]/foo.obj

#--- foo.s
.globl mainCRTStartup
mainCRTStartup:
  nop
