// RUN: %clang_cc1 -emit-pth %s -o %t
// RUN: %clang_cc1 -include-pth %t %s -E | grep 'file_to_include' | count 2
#include "file_to_include.h"
