! RUN: %flang -print-resource-dir -resource-dir=%S/Inputs/resource_dir \
! RUN:  | FileCheck -check-prefix=PRINT-RESOURCE-DIR -DFILE=%S/Inputs/resource_dir %s
! PRINT-RESOURCE-DIR: [[FILE]]
