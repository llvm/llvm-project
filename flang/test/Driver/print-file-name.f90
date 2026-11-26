! Test that -print-file-name finds the correct file.

! RUN: %flang -print-file-name=share/asan_ignorelist.txt \
! RUN:     -resource-dir=%S/Inputs/resource_dir \
! RUN:     --target=x86_64-unknown-linux-gnu 2>&1 \
! RUN:   | FileCheck --check-prefix=CHECK-RESOURCE-DIR %s
! CHECK-RESOURCE-DIR: resource_dir{{/|\\}}share{{/|\\}}asan_ignorelist.txt

! RUN: %flang -print-file-name=libflang_rt.runtime.a \
! RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
! RUN:     --target=x86_64-unknown-linux-gnu 2>&1 \
! RUN:   | FileCheck --check-prefix=CHECK-FLANG-RT %s
! CHECK-FLANG-RT: resource_dir_with_per_target_subdir{{/|\\}}lib{{/|\\}}x86_64-unknown-linux-gnu{{/|\\}}libflang_rt.runtime.a
