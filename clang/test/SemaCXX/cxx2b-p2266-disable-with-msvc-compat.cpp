// RUN: %clang_cc1 -std=c++2b -fsyntax-only -fcxx-exceptions                    -verify=new %s
// RUN: %clang_cc1 -std=c++2b -fsyntax-only -fcxx-exceptions -fms-compatibility -verify=old %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fcxx-exceptions                    -verify=old %s

// FIXME: This is a test for a temporary workaround where we disable simpler implicit moves
//        when compiling with -fms-compatibility, because the MSVC STL does not compile.
//        A better workaround is under discussion.
//        The test cases here are just a copy from `CXX/class/class.init/class.copy.elision/p3.cpp`,
//        so feel free to delete this file when the workaround is not needed anymore.

struct CopyOnly {
  CopyOnly(); // new-note {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
  // new-note@-1 {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
  CopyOnly(CopyOnly &); // new-note {{candidate constructor not viable: expects an lvalue for 1st argument}}
  // new-note@-1 {{candidate constructor not viable: expects an lvalue for 1st argument}}
};
struct MoveOnly {
  MoveOnly();
  MoveOnly(MoveOnly &&);
};
MoveOnly &&rref();

MoveOnly &&test1(MoveOnly &&w) {
  return w; // old-error {{cannot bind to lvalue of type}}
}

CopyOnly test2(bool b) {
  static CopyOnly w1;
  CopyOnly w2;
  if (b) {
    return w1;
  } else {
    return w2; // new-error {{no matching constructor for initialization}}
  }
}

template <class T> T &&test3(T &&x) { return x; } // old-error {{cannot bind to lvalue of type}}
template MoveOnly &test3<MoveOnly &>(MoveOnly &);
template MoveOnly &&test3<MoveOnly>(MoveOnly &&); // old-note {{in instantiation of function template specialization}}

MoveOnly &&test4() {
  MoveOnly &&x = rref();
  return x; // old-error {{cannot bind to lvalue of type}}
}

void test5() try {
  CopyOnly x;
  throw x; // new-error {{no matching constructor for initialization}}
} catch (...) {
}
