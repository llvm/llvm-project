! REQUIRES: x86-registered-target

! The driver forwards the companion path to the frontend.
! RUN: %flang -target x86_64-unknown-linux-gnu -flto=thin -fthin-link-bitcode=%t.min.bc -c %s -o %t.o -### 2>&1 | FileCheck %s --check-prefix=DRIVER
! DRIVER: "-fc1"
! DRIVER-SAME: "-flto=thin"
! DRIVER-SAME: "-fthin-link-bitcode={{.*}}.min.bc"

! Both outputs come from one ThinLTO compile. The minimized output has a
! summary and the full output's module hash, but no function bodies or debug info.
! RUN: %flang -target x86_64-unknown-linux-gnu -g -flto=thin -fthin-link-bitcode=%t.min.bc -c %s -o %t.o
! RUN: llvm-dis %t.o -o - | FileCheck %s --check-prefix=FULL-IR
! RUN: llvm-bcanalyzer -dump %t.o > %t.full.dump
! RUN: llvm-bcanalyzer -dump %t.min.bc > %t.min.dump
! RUN: FileCheck %s --check-prefix=FULL-BC < %t.full.dump
! RUN: FileCheck %s --check-prefix=MIN-BC < %t.min.dump
! RUN: %if llvm-lto %{ llvm-lto -thinlto-action=thinlink -o %t.index.bc %t.min.bc %}
! RUN: %if llvm-lto %{ llvm-dis %t.index.bc -o - | FileCheck %s --check-prefix=INDEX %}
! RUN: cat %t.full.dump %t.min.dump | FileCheck %s --check-prefix=HASH
! FULL-IR: define{{.*}} @fortran_function()
! FULL-BC: <COMPILE_UNIT
! FULL-BC: <GLOBALVAL_SUMMARY_BLOCK
! MIN-BC-NOT: <COMPILE_UNIT
! MIN-BC-NOT: <FUNCTION_BLOCK
! MIN-BC: <GLOBALVAL_SUMMARY_BLOCK
! MIN-BC-NOT: <COMPILE_UNIT
! MIN-BC-NOT: <FUNCTION_BLOCK
! INDEX: ^{{[0-9]+}} = gv: (guid:
! HASH: <HASH op0=[[H0:[0-9]+]] op1=[[H1:[0-9]+]] op2=[[H2:[0-9]+]] op3=[[H3:[0-9]+]] op4=[[H4:[0-9]+]]/>
! HASH: <HASH op0=[[H0]] op1=[[H1]] op2=[[H2]] op3=[[H3]] op4=[[H4]]/>

! Match Clang: the option only writes a companion for ThinLTO bitcode.
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -fthin-link-bitcode=%t.no-lto.bc -emit-llvm-bc %s -o %t.no-lto.o
! RUN: not test -e %t.no-lto.bc
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -flto=full -fthin-link-bitcode=%t.full-lto.bc -emit-llvm-bc %s -o %t.full-lto.o
! RUN: not test -e %t.full-lto.bc
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -flto=thin -fthin-link-bitcode=%t.assembly.bc -S %s -o %t.s
! RUN: not test -e %t.assembly.bc

! A failed companion open reports the path and removes the primary output.
! RUN: not %flang_fc1 -triple x86_64-unknown-linux-gnu -flto=thin -fthin-link-bitcode=%t.missing/companion.bc -emit-llvm-bc %s -o %t.failed.bc 2>&1 | FileCheck %s --check-prefix=ERROR
! RUN: not test -e %t.failed.bc
! ERROR: error: unable to open output file '{{.*}}.missing{{.}}companion.bc':

subroutine fortran_function() bind(c)
end subroutine
