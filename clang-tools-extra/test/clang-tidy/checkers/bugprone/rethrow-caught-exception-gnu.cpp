// RUN: %check_clang_tidy -std=gnu++17 %s bugprone-rethrow-caught-exception %t -- -- -fexceptions

struct E {};

void mayThrow();
void consume(int);

// A variable-length array bound is evaluated at runtime in the handler.
void vlaBoundWarns() {
  try {
    mayThrow();
  } catch (const E &Err) {
    int Count[(throw Err, 2)];
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: throwing a copy of the caught 'E' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: int Count[(throw, 2)];
    consume(sizeof(Count));
  }
}

void vlaTypedefWarns() {
  try {
    mayThrow();
  } catch (const E &Err) {
    typedef int Chunk[(throw Err, 2)];
    // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: throwing a copy of the caught 'E' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: typedef int Chunk[(throw, 2)];
    consume(sizeof(Chunk));
  }
}

void vlaAliasWarns() {
  try {
    mayThrow();
  } catch (const E &Err) {
    using Chunk = int[(throw Err, 2)];
    // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: throwing a copy of the caught 'E' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: using Chunk = int[(throw, 2)];
    consume(sizeof(Chunk));
  }
}

// An array bound nested inside `__typeof__` still executes at runtime when
// the array is variably modified.
void vlaTypeofBoundWarns() {
  try {
    mayThrow();
  } catch (const E &Err) {
    typedef __typeof__(int[(throw Err, 2)]) Chunk;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: throwing a copy of the caught 'E' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: typedef __typeof__(int[(throw, 2)]) Chunk;
    consume(sizeof(Chunk));
  }
}

// Throws inside `sizeof` type operands are not traversed by matchers, so
// they are never flagged, even when a variably-modified bound would execute
// them at runtime.
void sizeofVlaIsUnreached() {
  try {
    mayThrow();
  } catch (const E &Err) {
    consume(sizeof(int[(throw Err, 2)]));
  }
}
