# REQUIRES: x86
## Show that bitcode files and inputs to LTO-related options are stored in the
## reproduce package and that intermediate files (such as the LTO-compiled
## object) are not.

# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir/build1
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.dir/build1/foo.o
# RUN: echo > %t.dir/build1/empty_profile.txt
# RUN: cd %t.dir
# RUN: ld.lld build1/foo.o -o /dev/null --reproduce repro1.tar --lto-sample-profile=%t.dir/build1/empty_profile.txt
# RUN: tar tvf repro1.tar | FileCheck %s --implicit-check-not={{.}}

# CHECK-DAG: {{.*}} repro1/{{.*}}/empty_profile.txt
# CHECK-DAG: {{.*}} repro1/response.txt
# CHECK-DAG: {{.*}} repro1/version.txt
# CHECK-DAG: {{.*}} repro1/{{.*}}/foo.o
