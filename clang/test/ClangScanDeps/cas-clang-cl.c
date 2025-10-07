// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cl -c /clang:-fdepscan=inline /clang:-fdepscan-include-tree -Xclang -fcas-path -Xclang %t/cas -Xclang -Rcompile-job-cache -- %t/test.c 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: %clang_cl -c /clang:-fdepscan=inline /clang:-fdepscan-include-tree -Xclang -fcas-path -Xclang %t/cas -Xclang -Rcompile-job-cache -- %t/test.c 2>&1 | FileCheck %s -check-prefix=CACHE-HIT

// In debug mode
// RUN: %clang_cl -c /clang:-fdepscan=inline /clang:-fdepscan-include-tree -Xclang -fcas-path -Xclang %t/cas -Xclang -Rcompile-job-cache /Z7 -- %t/test.c 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: %clang_cl -c /clang:-fdepscan=inline /clang:-fdepscan-include-tree -Xclang -fcas-path -Xclang %t/cas -Xclang -Rcompile-job-cache /Z7 -- %t/test.c 2>&1 | FileCheck %s -check-prefix=CACHE-HIT

// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS: remark: compile job cache miss

//--- test.c
int main() { return 0; }
