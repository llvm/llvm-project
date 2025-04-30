#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: env KEEP_FILES=%keep FLAGS=%flags TEST_SRC=%/s MAKE_FILE_DIR=%/S/.. bash %/S/runmake | tee %/t
# RUN: cat %t | FileCheck %S/runmake
