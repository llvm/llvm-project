/* This test checks the handling of YAML profile with different block orders.
# RUN: %clang %cflags %s -o %t.exe
# RUN: link_fdata %s %t.exe %t.fdata
# RUN: llvm-bolt %t.exe -o %t.null -data %t.fdata -w %t.yaml
# RUN: FileCheck %s --input-file %t.yaml --check-prefix=CHECK-BINARY
# CHECK-BINARY: dfs-order: false
# RUN: llvm-bolt %t.exe -o %t.null -data %t.fdata -w %t.yaml --profile-use-dfs
# RUN: FileCheck %s --input-file %t.yaml --check-prefix=CHECK-DFS
# CHECK-DFS: dfs-order: true

# FDATA: 0 [unknown] 0 1 main 0 0 0
*/
int main() { return 0; }
