# Call to "size" intrinsic is inlined
#
# Shared lit script for each tests. Run bash commands that run tests with make.

# RUN: env KEEP_FILES=%keep FLAGS=%flags TEST_SRC=%/s MAKE_FILE_DIR=%/S/.. bash %/S/runmake | tee %/t
# RUN: cat %t | FileCheck %S/runmake
