// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 -Wthread-safety %s

// Without -Wthread-safety-beta the unchecked-result diagnostics cannot
// report a leaked try-acquire downstream, so a same-origin branch join
// keeps the eager lost-hold diagnosis instead of silently reconstituting
// the try-held state.

struct __attribute__((capability("mutex"))) Mutex {
  void Lock() __attribute__((acquire_capability()));
  void Unlock() __attribute__((release_capability()));
  bool TryLock() __attribute__((try_acquire_capability(true)));
};

Mutex mu;
int a __attribute__((guarded_by(mu)));
bool cond;

void same_origin_branch_join_warns_without_beta() {
  bool failed = !mu.TryLock(); // expected-note {{mutex acquired here}}
  if (failed)
    cond = true;
  a = 3;       // expected-warning {{mutex 'mu' is not held on every path through here}} \
               // expected-warning {{writing variable 'a' requires holding mutex 'mu' exclusively}}
  mu.Unlock(); // expected-warning {{releasing mutex 'mu' that was not held}}
}
