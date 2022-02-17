// RUN: %clang_cc1 %s -include %s
// RUN: %clang_cc1 %s -emit-pch -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch

// rdar://12239321 Make sure we don't emit a bogus
//     error: field designator 'e' does not refer to a non-static data member

#ifndef HEADER
#define HEADER
//===----------------------------------------------------------------------===//

struct U {
  union {
    struct {
      int e;
      int f;
    };

    int a;
  };
};

//===----------------------------------------------------------------------===//
#else
#if !defined(HEADER)
# error Header inclusion order messed up
#endif
//===----------------------------------------------------------------------===//

void bar(void) {
  static const struct U plan = { .e = 1 };
}

//===----------------------------------------------------------------------===//
#endif
