// RUN: %clang -print-resource-dir --target=x86_64-unknown-linux-gnu \
// RUN:   -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck -check-prefix=PRINT-RESOURCE-DIR -DFILE=%S/Inputs/resource_dir %s
// PRINT-RESOURCE-DIR: [[FILE]]
