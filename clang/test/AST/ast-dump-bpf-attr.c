// RUN: %clang_cc1 -triple bpf-pc-linux-gnu -ast-dump  %s \
// RUN: | FileCheck --strict-whitespace %s

int __attribute__((btf_type_tag("rcu"))) * g;
// CHECK: VarDecl{{.*}}g 'int  __attribute__((btf_type_tag("rcu")))*'
