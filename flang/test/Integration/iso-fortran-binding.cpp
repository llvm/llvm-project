// UNSUPPORTED: system-windows
// RUN: split-file %s %t
// RUN: chmod +x %t/runtest.sh
// RUN: %t/runtest.sh %t %t/cppfile.cpp %flang | FileCheck %s

//--- cppfile.cpp
extern "C" {
#include "ISO_Fortran_binding.h"
}
#include <iostream>

int main() {
  std::cout << "PASS\n";
  return 0;
}

// CHECK: PASS
// clang-format off
//--- runtest.sh
#!/bin/bash
TMPDIR=$1
CPPFILE=$2
FLANG=$3
BINDIR=`dirname $FLANG`
CPPCOMP=$BINDIR/clang++
if [ -x $CPPCOMP ]
then
  $CPPCOMP $CPPFILE -o $TMPDIR/a.out
  $TMPDIR/a.out # should print "PASS"
else
  # No clang compiler, just pass by default
  echo "PASS"
fi
