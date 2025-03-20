// RUN: %clang -### %s 2>&1 \
// RUN:     --target=aarch64-unknown-linux-musl \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_empty \
// RUN:     --sysroot=%S/Inputs/musl_sysroot_with_builtins \
// RUN:     -static \
// RUN:     -rtlib=compiler-rt \
// RUN:   | FileCheck %s

// CHECK-DAG: "--sysroot={{[^"]+}}{{/|\\\\}}Inputs{{/|\\\\}}musl_sysroot_with_builtins"
// CHECK-DAG: "{{[^"]+}}{{/|\\\\}}Inputs{{/|\\\\}}musl_sysroot_with_builtins{{/|\\\\}}lib{{/|\\\\}}libclang_rt.builtins.a"
