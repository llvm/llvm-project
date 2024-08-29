// RUN: split-file %s %t
// RUN: sed -e "s@FROM_DIR@%{/t:regex_replacement}/From@g" -e "s@TO_DIR@%{/t:regex_replacement}/Fake@g" %S/Inputs/vfsoverlay-directory-remap.yaml > %t/to-fake.yaml
// RUN: sed -e "s@FROM_DIR@%{/t:regex_replacement}/Fake@g" -e "s@TO_DIR@%{/t:regex_replacement}/To@g" %S/Inputs/vfsoverlay-directory-remap.yaml > %t/from-fake.yaml

// RUN: %clang_cc1 -Werror -fno-pch-timestamp -fvalidate-ast-input-files-content -ivfsoverlay %t/to-fake.yaml -emit-pch -o "%t.pch" %t/From/../From/B.h

// Remove the `From` directory as we don't want to accidentally find that source if the PCH hasn't remapped using the VFS!
// RUN: rm -rf %t/From

// The PCH will be invalid because the `Fake` directory does not exist.
// RUN: not %clang_cc1 -fno-pch-timestamp -fvalidate-ast-input-files-content -verify-pch %t.pch

// But if we specify the correct VFS overlay it'll verify clean.
// RUN: %clang_cc1 -fno-pch-timestamp -fvalidate-ast-input-files-content -verify-pch -ivfsoverlay %t/from-fake.yaml %t.pch

// RUN: %clang_cc1 -fno-pch-timestamp -fvalidate-ast-input-files-content -Werror -I %t/To -ivfsoverlay %t/from-fake.yaml -include-pch "%t.pch" -emit-llvm -C %t/Test.c -o %t.o

//--- From/B.h
#pragma once
int SomeFunc() { return 13; }
//--- To/B.h
#pragma once
int SomeFunc() { return 13; }
//--- Test.c
#include "B.h"

int UseSomeFunc() { return SomeFunc(); }
