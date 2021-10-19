// RUN: %clang_cc1 -std=c++2b -fsyntax-only                    -verify=cxx2b,new %s
// RUN: %clang_cc1 -std=c++2b -fsyntax-only -fms-compatibility -verify=cxx2b,old %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only                    -verify=cxx20,old %s

// FIXME: This is a test for a temporary workaround where we disable simpler implicit moves
//        in the STL when compiling with -fms-compatibility, because of issues with the
//        implementation there.
//        Feel free to delete this file when the workaround is not needed anymore.

#if __INCLUDE_LEVEL__ == 0

#if __cpluscplus > 202002L && __cpp_implicit_move < 202011L
#error "__cpp_implicit_move not defined correctly"
#endif

struct nocopy {
  nocopy(nocopy &&);
};

int &&mt1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &mt2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
nocopy mt3(nocopy x) { return x; }

namespace {
int &&mt1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &mt2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
nocopy mt3(nocopy x) { return x; }
} // namespace

namespace foo {
int &&mt1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &mt2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
namespace std {
int &&mt1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &mt2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
nocopy mt3(nocopy x) { return x; }
} // namespace std
} // namespace foo

namespace std {

int &&mt1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &mt2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
nocopy mt3(nocopy x) { return x; }

namespace {
int &&mt1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &mt2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
nocopy mt3(nocopy x) { return x; }
} // namespace

namespace foo {
int &&mt1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &mt2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
nocopy mt3(nocopy x) { return x; }
} // namespace foo

} // namespace std

#include __FILE__

#define SYSTEM
#include __FILE__

#elif !defined(SYSTEM)

int &&ut1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &ut2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
nocopy ut3(nocopy x) { return x; }

namespace {
int &&ut1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &ut2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
nocopy ut3(nocopy x) { return x; }
} // namespace

namespace foo {
int &&ut1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &ut2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
nocopy ut3(nocopy x) { return x; }
namespace std {
int &&ut1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &ut2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
nocopy ut3(nocopy x) { return x; }
} // namespace std
} // namespace foo

namespace std {

int &&ut1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &ut2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
nocopy ut3(nocopy x) { return x; }

namespace {
int &&ut1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &ut2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
nocopy ut3(nocopy x) { return x; }
} // namespace

namespace foo {
int &&ut1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &ut2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
nocopy ut3(nocopy x) { return x; }
} // namespace foo

} // namespace std

#else

#pragma GCC system_header

int &&st1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &st2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
nocopy st3(nocopy x) { return x; }

namespace {
int &&st1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &st2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
nocopy st3(nocopy x) { return x; }
} // namespace

namespace foo {
int &&st1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &st2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
nocopy st3(nocopy x) { return x; }
namespace std {
int &&st1(int &&x) { return x; } // cxx20-error {{cannot bind to lvalue}}
int &st2(int &&x) { return x; }  // cxx2b-error {{cannot bind to a temporary}}
nocopy st3(nocopy x) { return x; }
} // namespace std
} // namespace foo

namespace std {

int &&st1(int &&x) { return x; } // old-error {{cannot bind to lvalue}}
int &st2(int &&x) { return x; }  // new-error {{cannot bind to a temporary}}
nocopy st3(nocopy x) { return x; }

namespace {
int &&st1(int &&x) { return x; } // old-error {{cannot bind to lvalue}}
int &st2(int &&x) { return x; }  // new-error {{cannot bind to a temporary}}
nocopy st3(nocopy x) { return x; }
} // namespace

namespace foo {
int &&st1(int &&x) { return x; } // old-error {{cannot bind to lvalue}}
int &st2(int &&x) { return x; }  // new-error {{cannot bind to a temporary}}
nocopy st3(nocopy x) { return x; }
} // namespace foo

} // namespace std

#endif
