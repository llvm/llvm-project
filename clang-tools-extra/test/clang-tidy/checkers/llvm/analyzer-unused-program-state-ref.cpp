// RUN: %check_clang_tidy -std=c++17 %s llvm-analyzer-unused-program-state-ref %t

// Stub that mirrors clang::ento::ProgramStateRef without pulling in the
// analyzer headers. The check keys on the fully-qualified typedef name.
// The user-declared destructor makes it non-trivially destructible, exactly
// like the real IntrusiveRefCntPtr, so -Wunused-variable stays silent.
namespace clang {
namespace ento {
template <class T>
struct IntrusiveRefCntPtr {
  IntrusiveRefCntPtr();
  ~IntrusiveRefCntPtr();
};
class ProgramState;
typedef IntrusiveRefCntPtr<const ProgramState> ProgramStateRef;
} // namespace ento
} // namespace clang

using clang::ento::ProgramStateRef;

ProgramStateRef getState();
void use(ProgramStateRef State);

// Aggregate returned by value, decomposed via structured bindings, mirroring
// the analyzer's `std::pair<ProgramStateRef, ProgramStateRef>` from
// `ProgramState::assume()`.
struct StatePair {
  ProgramStateRef first;
  ProgramStateRef second;
};
struct MixedPair {
  ProgramStateRef first;
  int second;
};
StatePair assume();
MixedPair mixedAssume();

void unused_local() {
  ProgramStateRef State = getState();
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: unused 'ProgramStateRef' variable 'State' [llvm-analyzer-unused-program-state-ref]
}

void unused_no_init() {
  ProgramStateRef State;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: unused 'ProgramStateRef' variable 'State' [llvm-analyzer-unused-program-state-ref]
}

void used_read() {
  ProgramStateRef State = getState();
  use(State);
}

void parameter_is_ignored(ProgramStateRef State) {}

void maybe_unused_is_ignored() {
  [[maybe_unused]] ProgramStateRef State = getState();
}

void multi_all_unused() {
  ProgramStateRef A, B;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: unused 'ProgramStateRef' variable 'A' [llvm-analyzer-unused-program-state-ref]
  // CHECK-MESSAGES: :[[@LINE-2]]:22: warning: unused 'ProgramStateRef' variable 'B' [llvm-analyzer-unused-program-state-ref]
}

void multi_mixed() {
  ProgramStateRef A, B = getState();
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: unused 'ProgramStateRef' variable 'A' [llvm-analyzer-unused-program-state-ref]
  use(B);
}

#define DECLARE_STATE ProgramStateRef State = getState();
void from_macro() {
  DECLARE_STATE
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: unused 'ProgramStateRef' variable 'State' [llvm-analyzer-unused-program-state-ref]
}

void structured_binding_all_unused() {
  auto [StTrue, StFalse] = assume();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: unused 'ProgramStateRef' structured binding [llvm-analyzer-unused-program-state-ref]
}

void structured_binding_partially_used() {
  // A partially-used decomposition is out of scope; leave it alone.
  auto [StTrue, StFalse] = assume();
  use(StTrue);
}

void structured_binding_maybe_unused() {
  [[maybe_unused]] auto [StTrue, StFalse] = assume();
}

void structured_binding_non_program_state_ref() {
  // Not all bindings are ProgramStateRef; out of scope.
  auto [State, Count] = mixedAssume();
  use(State);
}

