! DEFINE: %{resource_dir} = %S/Inputs/resource_dir
! RUN: %flang -print-resource-dir -resource-dir=%{resource_dir}.. \
! RUN:  | FileCheck -check-prefix=PRINT-RESOURCE-DIR -DFILE=%{resource_dir} %s
! PRINT-RESOURCE-DIR: [[FILE]]
