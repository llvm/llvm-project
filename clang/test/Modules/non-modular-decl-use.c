// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -I %t/include %t/test.c \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// Test when a decl is present in multiple modules through an inclusion of
// a non-modular header. Make sure such decl is serialized correctly and can be
// used after deserialization.

//--- include/non-modular.h
#ifndef NON_MODULAR_H
#define NON_MODULAR_H

union TestUnion {
  int x;
  float y;
};

struct ReferenceUnion1 {
  union TestUnion name;
  unsigned versionMajor;
};
struct ReferenceUnion2 {
  union TestUnion name;
  unsigned versionMinor;
};

// Test another kind of RecordDecl.
struct TestStruct {
  int p;
  float q;
};

struct ReferenceStruct1 {
  unsigned fieldA;
  struct TestStruct fieldB;
};

struct ReferenceStruct2 {
  unsigned field1;
  struct TestStruct field2;
};

#endif

//--- include/piecewise1-empty.h
//--- include/piecewise1-initially-hidden.h
#include <non-modular.h>

//--- include/piecewise2-empty.h
//--- include/piecewise2-initially-hidden.h
#include <non-modular.h>

//--- include/with-multiple-decls.h
#include <piecewise1-empty.h>
// Include the non-modular header and resolve a name duplication between decl
// in non-modular.h and in piecewise1-initially-hidden.h, create a
// redeclaration chain.
#include <non-modular.h>
// Include a decl with a duplicate name again to add more to IdentifierResolver.
#include <piecewise2-empty.h>

//--- include/module.modulemap
module Piecewise1 {
  module Empty {
    header "piecewise1-empty.h"
  }
  module InitiallyHidden {
    header "piecewise1-initially-hidden.h"
    export *
  }
}

module Piecewise2 {
  module Empty {
    header "piecewise2-empty.h"
  }
  module InitiallyHidden {
    header "piecewise2-initially-hidden.h"
    export *
  }
}

module WithMultipleDecls {
  header "with-multiple-decls.h"
  export *
}

//--- test.c
#include <with-multiple-decls.h>

struct Test {
  int x;
  union TestUnion name;
};

struct Test2 {
  struct TestStruct name;
  float y;
};
