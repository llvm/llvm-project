// RUN: %check_clang_tidy -std=c++20-or-later %s bugprone-rethrow-caught-exception %t -- -- -fexceptions

struct BaseError {};

void mayThrow();
void consume(int);

struct NonCopyableError {
  NonCopyableError() = default;
  NonCopyableError(const NonCopyableError &) = delete;
};

// A `requires` operand is never evaluated: rewriting it to a bare `throw;`
// could flip the requirement (false for the non-copyable type below).
template <class T> bool requirementHolds() {
  try {
    mayThrow();
  } catch (const T &Err) {
    return requires { throw Err; };
  }
  return false;
}

void instantiateRequirement() {
  consume(requirementHolds<NonCopyableError>() ? 1 : 0);
  consume(requirementHolds<BaseError>() ? 1 : 0);
}

void captureInitIsNotTraversed() {
  // Lambda capture initializers run immediately in the handler, but matcher
  // traversal never visits them, so no warning is produced here. If that
  // ever changes, this should warn with a fix to a bare `throw;`.
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    auto Fn = [Count = (throw Err, 0)] {};
    (void)Fn;
  }
}
