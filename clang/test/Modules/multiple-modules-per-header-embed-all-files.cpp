// RUN: rm -rf %t
// RUN: split-file %s %t
//
// First, build a module with a header.
//
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%t/modules1.map -fmodule-name=a -emit-module -xc++ -fmodules-embed-all-files -o %t/a.pcm %t/modules1.map
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%t/modules1.map -fmodule-map-file=%t/modules2.map -fmodule-name=b -emit-module \
// RUN:   -fmodule-file=%t/a.pcm -xc++ -fmodules-embed-all-files -o %t/b.pcm %t/modules2.map
// 
// Remove the header and check we can still build the code that finds it in a PCM.
//
// RUN: rm %t/foo.h
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%t/modules2.map -fmodule-file=%t/b.pcm -fsyntax-only %t/use.cpp

//--- modules1.map
module a {
  module foo {
    header "foo.h"
    export *
  }
  export *
}

//--- modules2.map
module all_textual {
  module foo {
    textual header "foo.h"
    export *
  }
  module wrap_foo {
    textual header "wrap_foo.h"
    export *
  }
  export *
}

module b {
  module wrap_foo {
    private header "wrap_foo.h"
    export *
  }
  export *
}


//--- foo.h
#ifndef FOO_H
#define FOO_H
void foo();
#endif

//--- wrap_foo.h
#include "foo.h"

//--- use.cpp
#include "wrap_foo.h"

void test() {
  foo();
}
