// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-exception-escape %t -- \
// RUN:     -config="{CheckOptions: { \
// RUN:         bugprone-exception-escape.CheckDestructors: false, \
// RUN:         bugprone-exception-escape.CheckMoveMemberFunctions: false, \
// RUN:         bugprone-exception-escape.CheckMain: false, \
// RUN:         bugprone-exception-escape.CheckedSwapFunctions: '', \
// RUN:         bugprone-exception-escape.CheckNothrowFunctions: false \
// RUN:     }}" \
// RUN:     -- -fexceptions

// CHECK-MESSAGES-NOT: warning:

struct destructor {
  ~destructor() {
    throw 1;
  }
};

struct move {
    move(move&&) { throw 42; }
    move& operator=(move&&) { throw 42; }
};

void swap(int&, int&) {
  throw 1;
}

void iter_swap(int&, int&) {
  throw 1;
}

void iter_move(int&) {
  throw 1;
}

void nothrow_func() throw() {
  throw 1;
}

void noexcept_func() noexcept {
  throw 1;
}

int main() {
  throw 1;
  return 0;
}
