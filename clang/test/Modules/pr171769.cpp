// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t
//

// RUN: %clang_cc1 -fmodule-name=A -fno-cxx-modules -xc++ -emit-module \
// RUN:   -fmodules A.cppmap -o A.pcm
// RUN: %clang_cc1 -fmodule-name=B -fno-cxx-modules -xc++ -emit-module \
// RUN:   -fmodules -fmodule-file=A.pcm B.cppmap -o B.pcm
// RUN: %clang_cc1 -fmodule-name=C -fno-cxx-modules -xc++ -emit-module \
// RUN:   -fmodules C.cppmap -o C.pcm
// RUN: %clang_cc1 -fmodule-name=D -fno-cxx-modules -xc++ -emit-module \
// RUN:   -fmodules -fmodule-file=C.pcm D.cppmap -o D.pcm
// RUN: %clang_cc1 -fmodule-name=E -fno-cxx-modules -xc++ -emit-module \
// RUN:   -fmodules -fmodule-file=B.pcm E.cppmap -o E.pcm
// RUN: %clang_cc1 -fmodule-name=F -fno-cxx-modules -xc++ -emit-module \
// RUN:   -fmodules -fmodule-file=D.pcm F.cppmap -o F.pcm
// RUN: %clang_cc1 -fmodule-name=G -fno-cxx-modules -xc++ -emit-module \
// RUN:   -fmodules -fmodule-file=E.pcm -fmodule-file=F.pcm G.cppmap -o G.pcm
// RUN: %clang_cc1 -fno-cxx-modules -fmodules -fmodule-file=G.pcm src.cpp \
// RUN:   -o /dev/null

//--- A.cppmap
module "A" { header "A.h" }

//--- A.h
int x;

//--- B.cppmap
module "B" {}

//--- C.cppmap
module "C" { header "C.h" }

//--- C.h
namespace xyz {}

//--- D.cppmap
module "D" {}

//--- E.cppmap
module "E" {}

//--- F.cppmap
module "F" { header "F.h" }

//--- F.h
namespace xyz { inline void func() {} }

//--- G.cppmap
module "G" { header "G.h" }

//--- G.h
#include "F.h"
namespace { void func2() { xyz::func(); } }

//--- hdr.h
#include "F.h"
namespace xyz_ns = xyz;

//--- src.cpp
#include "hdr.h"
