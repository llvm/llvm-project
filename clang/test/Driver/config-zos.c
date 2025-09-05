// Needs symlinks
// UNSUPPORTED: system-windows
// env -u is not supported on AIX.
// TODO(boomanaiden154): Remove this once we have switched over to lit's
// internal shell which does support env -u.
// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// REQUIRES: systemz-registered-target

// RUN: rm -rf %t && mkdir %t

// RUN: mkdir -p %t/testbin
// RUN: mkdir -p %t/etc
// RUN: ln -s %clang %t/testbin/clang
// RUN: echo "-DXYZ=789" >%t/etc/clang.cfg
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testbin/clang --target=s390x-ibm-zos -c -### -no-canonical-prefixes %s 2>&1 | FileCheck -DDIR=%t %s 
// RUN: env -u CLANG_NO_DEFAULT_CONFIG %t/testbin/clang --target=s390x-ibm-zos -c -### -no-canonical-prefixes --no-default-config %s 2>&1 | FileCheck -check-prefix=NOCONFIG %s 
//
// CHECK: Configuration file: [[DIR]]/etc/clang.cfg
// CHECK: "-D" "XYZ=789"
// NOCONFIG-NOT: Configuration file: {{.*}}/etc/clang.cfg
// NOCONFIG-NOT: "-D" "XYZ=789"
