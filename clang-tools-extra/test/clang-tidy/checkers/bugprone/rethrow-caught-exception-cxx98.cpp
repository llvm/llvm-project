// RUN: %check_clang_tidy -std=c++98 %s bugprone-rethrow-caught-exception %t -- -- -fexceptions

struct BaseError {};
struct OtherError {};
typedef BaseError TypedefError;
struct DefaultedCopy98 {
  int Value;
  explicit DefaultedCopy98(int V) : Value(V) {}
  DefaultedCopy98(const DefaultedCopy98 &Other, int = 0)
      : Value(Other.Value + 1) {}
};

void mayThrow();
void consume(int);

void rethrowConstRef98() {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void rethrowNonConstRef98() {
  try {
    mayThrow();
  } catch (BaseError &Err) {
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void rethrowScalar98() {
  try {
    mayThrow();
  } catch (const int &Err) {
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'int' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void rethrowTypedef98() {
  try {
    mayThrow();
  } catch (TypedefError &Err) {
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void bareThrowIsFine98() {
  try {
    mayThrow();
  } catch (const BaseError &) {
    throw;
  }
}

void catchAllIsFine98() {
  try {
    mayThrow();
  } catch (...) {
    throw;
  }
}

void catchByValueIsIgnored98(int Ignored) {
  try {
    mayThrow();
  } catch (BaseError Err) {
    throw Err;
  }
  consume(Ignored);
}

void unrelatedThrowIsIgnored98() {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    OtherError Fresh;
    consume(sizeof(Fresh) + sizeof(Err));
    throw Fresh;
  }
}

void explicitDefaultedParenIsIgnored98() {
  try {
    mayThrow();
  } catch (const DefaultedCopy98 &Err) {
    throw DefaultedCopy98(Err);
  }
}
