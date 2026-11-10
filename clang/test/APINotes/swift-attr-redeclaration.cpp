// An API-noted tag that is forward-declared before its definition must not
// accumulate duplicate 'swift_attr's on the definition.
// RUN: rm -rf %t && split-file %s %t
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -fapinotes-modules -I %t/Inputs -x c++ %t/test.cpp \
// RUN:   -ast-dump -ast-dump-filter FwdThenDefined \
// RUN:   | FileCheck --check-prefix=FWD --implicit-check-not=SwiftAttrAttr %s

// The forward declaration carries one copy of each annotation.
//
// FWD:      Dumping FwdThenDefined:
// FWD:      CXXRecordDecl {{.*}} struct FwdThenDefined
// FWD-NEXT:   SwiftAttrAttr {{.*}} "import_reference"
// FWD-NEXT:   SwiftAttrAttr {{.*}} "retain:FTDRetain"
// FWD-NEXT:   SwiftAttrAttr {{.*}} "release:FTDRelease"

// So must the definition: the copies it inherits and the copies API notes
// applies to it are the same three annotations.
//
// FWD:      Dumping FwdThenDefined:
// FWD:      CXXRecordDecl {{.*}} prev {{.*}} struct FwdThenDefined definition
// FWD:        SwiftAttrAttr {{.*}} "import_reference"
// FWD-NEXT:   SwiftAttrAttr {{.*}} "retain:FTDRetain"
// FWD-NEXT:   SwiftAttrAttr {{.*}} "release:FTDRelease"

//--- Inputs/module.modulemap
module Redecl {
  header "Redecl.h"
}

//--- Inputs/Redecl.apinotes
---
Name: Redecl
Tags:
- Name: FwdThenDefined
  SwiftImportAs: reference
  SwiftRetainOp: FTDRetain
  SwiftReleaseOp: FTDRelease

//--- Inputs/Redecl.h
struct FwdThenDefined;
struct FwdThenDefined {};

//--- test.cpp
#include "Redecl.h"
