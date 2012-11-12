// RUN: %clang_cc1 -fsyntax-only -Wheader-hygiene -verify %s

#ifdef BE_THE_HEADER
namespace warn_in_header_in_global_context {}
using namespace warn_in_header_in_global_context; // expected-warning {{using namespace directive in global context in header}}

// While we want to error on the previous using directive, we don't when we are
// inside a namespace
namespace dont_warn_here {
using namespace warn_in_header_in_global_context;
}

// We should warn in toplevel extern contexts.
namespace warn_inside_linkage {}
extern "C++" {
using namespace warn_inside_linkage; // expected-warning {{using namespace directive in global context in header}}
}

// This is really silly, but we should warn on it:
extern "C++" {
extern "C" {
extern "C++" {
using namespace warn_inside_linkage; // expected-warning {{using namespace directive in global context in header}}
}
}
}

// But we shouldn't warn in extern contexts inside namespaces.
namespace dont_warn_here {
extern "C++" {
using namespace warn_in_header_in_global_context;
}
}

// We also shouldn't warn in case of functions.
inline void foo() {
  using namespace warn_in_header_in_global_context;
}


namespace macronamespace {}
#define USING_MACRO using namespace macronamespace;

// |using namespace| through a macro should warn if the instantiation is in a
// header.
USING_MACRO // expected-warning {{using namespace directive in global context in header}}

#else

#define BE_THE_HEADER
#include __FILE__

namespace dont_warn {}
using namespace dont_warn;

// |using namespace| through a macro shouldn't warn if the instantiation is in a
// cc file.
USING_MACRO

#endif
