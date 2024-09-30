// REQUIRES: system-linux
// RUN: split-file %s %t
// RUN: chmod +x %t/mixed-runtest.sh
// RUN: %t/mixed-runtest.sh -stdlib=platform %t %t/mixed-cppfile.cpp
// %t/mixed-fortranfile.f90 %flang | FileCheck %s

//--- mixed-cppfile.cpp
#include <iostream>

extern "C" void hello(void) { std::cout << "Hello" << std::endl; }

// clang-format off
// CHECK: PASS
//--- mixed-fortranfile.f90
program mixed
  implicit none
  interface
    subroutine hello() bind(c)
      implicit none
    end subroutine
  end interface

  call hello()
end program

//--- mixed-runtest.sh
#!/bin/bash
LDFLAGS=$1
TMPDIR=$2
CPPFILE=$3
F90FILE=$4
FLANG=$5
BINDIR=`dirname $FLANG`
CPPCOMP=$BINDIR/clang++
if [ -x $CPPCOMP ]
then
  $CPPCOMP -shared $LDFLAGS $CPPFILE -o $TMPDIR/libmixed.so
  $FLANG $LDFLAGS -o $TMPDIR/mixed $F90FILE -L$TMPDIR -lmixed -Wl,-rpath=$TMPDIR
  if [ -x $TMPDIR/mixed ]
  then
    echo "PASS"
  else
    echo "FAIL"
  fi
else
  # No clang compiler, just pass by default
  echo "PASS"
fi
