// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 -Wthread-safety %s

// Without -Wthread-safety-beta the unchecked-result diagnostics cannot
// report a leaked try-acquire downstream, so a same-origin branch join
// keeps the eager lost-hold diagnosis instead of silently reconstituting
// the conditionally held state.

struct __attribute__((capability("mutex"))) Mutex {
  void Lock() __attribute__((acquire_capability()));
  void ReaderLock() __attribute__((acquire_shared_capability()));
  void Unlock() __attribute__((release_capability()));
  void ReaderUnlock() __attribute__((release_shared_capability()));
  bool TryLock() __attribute__((try_acquire_capability(true)));
  bool ReaderTryLock() __attribute__((try_acquire_shared_capability(true)));
};

Mutex mu;
int a __attribute__((guarded_by(mu)));
bool cond;
void g();
void needsNotHeld() __attribute__((locks_excluded(mu)));
void needsHeld() __attribute__((requires_capability(mu)));

void same_origin_branch_join_warns_without_beta() {
  bool failed = !mu.TryLock(); // expected-note {{mutex acquired here}}
  if (failed)
    cond = true;
  a = 3;       // expected-warning {{mutex 'mu' is not held on every path through here}} \
               // expected-warning {{writing variable 'a' requires holding mutex 'mu' exclusively}}
  mu.Unlock(); // expected-warning {{releasing mutex 'mu' that was not held}}
}

// Loop joins are the deliberate exception: the demotion applies there in
// either mode, so that gating it on beta cannot restore the false
// positives the exemption exists to remove -- the eager diagnosis at a
// loop join is not a leak report, it is a complaint about a join that
// loses nothing.
//
// The check-first idiom's continue latch does not reach that exemption
// yet: the loop-top check is decoded before the call is walked, so it
// resolves nothing and the body's success hold is reported against the
// loop's entry state (twice, from the pairwise intersection). Pinned in
// this mode so that a later commit's loop walk shows the change here.
void same_origin_continue_latch_still_conservative() {
  bool b = false;
  while (cond) { // expected-warning 2 {{expecting mutex 'mu' to be held at start of each loop}}
    if (b)
      mu.Unlock(); // expected-warning {{releasing mutex 'mu' that was not held}}
    b = mu.TryLock(); // expected-note 2 {{mutex acquired here}}
    if (!b)
      continue;
    a = 1;
  }
  if (b)
    mu.Unlock();
}

// The same shape leaking (nothing releases after the loop) reports the same
// way, and no more: the loop-carried result leaves no fact for the
// unchecked-result diagnostics to report, in either mode. Pinned so that a
// future beta-only leak report is a visible change rather than an accident.
void same_origin_continue_latch_leak_adds_nothing() {
  bool b = false;
  while (cond) { // expected-warning 2 {{expecting mutex 'mu' to be held at start of each loop}}
    if (b)
      mu.Unlock(); // expected-warning {{releasing mutex 'mu' that was not held}}
    b = mu.TryLock(); // expected-note 2 {{mutex acquired here}}
    if (!b)
      continue;
    a = 1;
  }
}

// The same-origin demotion of a level, not of a whole hold: a try-acquire
// over the capability's own hold leaves the success path one level deeper,
// which under beta demotes to the call's conditional try fact (the leak is
// reported there as an unchecked result). Without beta the join diagnoses
// the depth eagerly and the deeper fact wins, so the release below unwinds
// only the extra level and the base is still held at the end.
void same_origin_depth_pair_warns_without_beta() {
  mu.Lock();   // expected-note 2 {{mutex acquired here}}
  if (mu.TryLock()) {
  }
  mu.Unlock(); // expected-warning {{mutex 'mu' is not held on every path through here with equal reentrancy depth}}
} // expected-warning {{mutex 'mu' is still held at the end of function}}

// The idioms the same-origin demotion is for, in the mode where it does not
// apply at branch merges: each keeps the eager diagnosis the merge gives.

// Two try-acquires of one capability, each branched on (the merged state is
// still two conditional try facts, so the uses below are diagnosed).
void two_calls_each_branched() {
  if (mu.TryLock()) // expected-note {{mutex acquired here}}
    cond = true;
  if (mu.TryLock()) // expected-warning {{mutex 'mu' is not held on every path through here}} \
                    // expected-note {{mutex acquired here}}
    cond = true;
  a = 1;       // expected-warning {{writing variable 'a' requires holding mutex 'mu' exclusively}} \
               // expected-warning {{mutex 'mu' is not held on every path through here}}
  mu.Unlock(); // expected-warning {{releasing mutex 'mu' that was not held}}
}

// A shared try-acquire takes the rebranch exemption like an exclusive one,
// which is not gated on beta: the merge before the call is silent in this
// mode too.
void shared_same_origin_join() {
  bool ok = mu.ReaderTryLock();
  if (ok)
    cond = true;
  g();
  if (ok)
    mu.ReaderUnlock();
}

// The negative requirement of a capability left conditionally held by a
// same-origin merge is still violated: a possible hold is not "not held".
void negative_requirement_after_join() {
  bool ok = mu.TryLock();
  if (ok)
    cond = true;
  needsNotHeld(); // expected-warning {{cannot call function 'needsNotHeld' while mutex 'mu' may be held}}
  if (ok)
    mu.Unlock();
}

// Nor does a merged conditional hold satisfy a positive requirement.
void positive_requirement_after_join() {
  bool ok = mu.TryLock();
  if (ok)
    cond = true;
  needsHeld(); // expected-warning {{calling function 'needsHeld' requires holding mutex 'mu' exclusively}}
  if (ok)
    mu.Unlock();
}

// The end-of-function comparison is not a merge the exemption applies to:
// the declared expected set has no try facts to demote against.
void leaves_conditional_at_end() __attribute__((requires_capability(mu))) { // expected-note {{mutex acquired here}}
  mu.Unlock();
  bool ok = mu.TryLock(); // expected-note {{mutex acquired here}}
  if (ok)
    cond = true;
} // expected-warning {{mutex 'mu' is not held on every path through here}} \
  // expected-warning {{expecting mutex 'mu' to be held at the end of function}}

// The check-first idiom in its if/else and goto spellings: correct
// programs, and silent in this mode as under beta -- the loop-top check
// resolves nothing either way, so there is no eager diagnosis to lose.
void check_first_if_else() {
  bool b = false;
  while (cond) {
    if (b) {
    } else {
      b = mu.TryLock();
    }
  }
  if (b)
    mu.Unlock();
}

void check_first_goto() {
  bool b = false;
top:
  if (cond) {
    if (b)
      goto top;
    b = mu.TryLock();
    goto top;
  }
  if (b)
    mu.Unlock();
}

// The release-and-reset loop keeps the conservative in-loop report, in both
// modes: the loop-top check does not resolve, so the release it guards is
// diagnosed. Correct code, and the residue the loop walk is to remove.
void release_and_reset() {
  bool b = false;
  while (cond) {
    if (b) {
      mu.Unlock(); // expected-warning {{releasing mutex 'mu' that was not held}}
      b = false;
      continue;
    }
    b = mu.TryLock();
  }
  if (b)
    mu.Unlock();
}

// The in-loop acquire-and-continue leak, and a release that follows the
// re-acquire without resetting the flag: both report without beta.
void in_loop_acquire_continue() {
  while (cond) { // expected-warning {{expecting mutex 'mu' to be held at start of each loop}}
    if (mu.TryLock()) { // expected-note {{mutex acquired here}}
      a = 1;
      continue;
    }
    g();
  }
  mu.Unlock(); // expected-warning {{releasing mutex 'mu' that was not held}}
}

void release_after_reacquire() {
  bool locked = false;
  while (cond) {
    if (locked)
      a = a + 1; // expected-warning {{writing variable 'a' requires holding mutex 'mu' exclusively}} \
                 // expected-warning {{reading variable 'a' requires holding mutex 'mu'}}
    locked = mu.TryLock();
    if (locked) {
      a = 0;
      mu.Unlock();
    }
  }
}
