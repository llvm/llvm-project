// RUN: %clang_cc1 -fsyntax-only -std=c++17 -flifetime-safety-inference \
// RUN:   -fexperimental-lifetime-safety-tu-analysis \
// RUN:   -Wlifetime-safety-suggestions -Wno-dangling -verify %s

// Verify that parenthesised declarator names with trailing return types don't
// crash the lifetimebound fixit suggestion logic.  getLifetimeBoundFixIt() for
// CXXMethodDecl used TypeLoc::getAs<FunctionTypeLoc>() to navigate into the
// type, but parenthesised declarators (e.g. 'auto (f)() & -> T') wrap the
// FunctionTypeLoc inside a ParenTypeLoc which getAs<> cannot see through.
//
// https://github.com/llvm/llvm-project/issues/237332

struct S {
  // Crash: parenthesised operator name + trailing return type.
  auto (operator++)() & -> S & { return *this; } // expected-warning {{implicit this in intra-TU function should be marked [[clang::lifetimebound]]}}
                                                  // expected-note@-1 {{param returned here}}

  // Parenthesised operator name, conventional (non-trailing) return type.
  S & (operator--)(int) { return *this; } // expected-warning {{implicit this in intra-TU function should be marked [[clang::lifetimebound]]}}
                                          // expected-note@-1 {{param returned here}}

  // Non-parenthesised trailing return -- was already working, must stay so.
  auto operator++(int) -> S & { return *this; } // expected-warning {{implicit this in intra-TU function should be marked [[clang::lifetimebound]]}}
                                                 // expected-note@-1 {{param returned here}}
};
