// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 -Wthread-safety -Wthread-safety-beta %s

// A try-acquire call's resolved result is a try fact of its own, separate
// from the definite hold it proved: what happens to the hold afterwards
// (another call acquires the capability, a release releases the result)
// leaves the record of the call's result standing, and one call's record
// says nothing about another call's.

struct __attribute__((capability("mutex"))) Mutex {
  void Lock() __attribute__((acquire_capability()));
  void Unlock() __attribute__((release_capability()));
  bool TryLock() __attribute__((try_acquire_capability(true)));
};

struct __attribute__((capability("mutex"), reentrant_capability)) RMutex {
  void Lock() __attribute__((acquire_capability()));
  void Unlock() __attribute__((release_capability()));
  bool TryLock() __attribute__((try_acquire_capability(true)));
};

Mutex mu;
RMutex rmu;
int a __attribute__((guarded_by(mu)));
int b __attribute__((guarded_by(rmu)));
bool cond;

// The call's failure record survives a later acquisition by a blocking
// call: the result is still falsy, so a re-check inside the locked region
// finds its success edge infeasible and the dead arm's release resolves
// nothing at the join. (A negative fact consumed by the acquisition could
// not carry the failure, and the dead arm then drew a lost-hold, a
// guarded-write and a stray-release warning.)
void failed_survives_lock() {
  bool ok = mu.TryLock();
  if (!ok) {
    mu.Lock();
    if (ok) {
      mu.Unlock();
    }
    a = 1;
  }
  mu.Unlock();
}

// A blocking acquisition retires the call's released record: the
// capability is held again, through an acquisition of its own, so there is
// no lost hold for the record to protect. The branch on the stale result
// still resurrects nothing -- the hold it would re-materialize is refuted
// by the blocking call's -- and does not release that hold either.
// (Kept past the re-acquisition, the record made a release inside a loop
// that re-acquires look, at the loop's exit, like a release the loop left
// standing: tryheld_loop_release_then_reacquire.)
void spent_retired_by_lock() {
  bool ok = mu.TryLock();
  if (ok)
    mu.Unlock();
  mu.Lock();
  if (ok) {
    a = 1;
  }
  mu.Unlock();
}

// Two proved holds over one reentrant capability, one level released:
// definite levels are interchangeable, so both try facts still prove the
// remaining level and the second release leaves the capability released
// on that path. The hold is lost once at the three-way join, and the
// second call's result reaches the end of the function unchecked.
void two_proofs_one_release() {
  bool ok1 = rmu.TryLock(); // expected-note {{mutex acquired here}}
  bool ok2 = rmu.TryLock(); // expected-note {{mutex acquired here}}
  if (ok1 && ok2) {
    rmu.Unlock();
    if (ok1) {
      b = 1;
      rmu.Unlock();
    }
  }
} // expected-warning {{mutex 'rmu' is not held on every path through here}} \
  // expected-warning {{unchecked result of try-acquire; mutex 'rmu' may still be held at the end of function}}

// The stale-result veto is per call: another call's release on the other
// side of a join says nothing about this call's result, so the hold this
// call proved is still carried, demoted to its conditional try fact, into
// the rebranch on its stored result. (A single negative fact per
// capability could not tell whose stale truth would do the resurrecting,
// and refused the demotion: the hold was lost at the join, the guarded
// write and the release under the rebranch both warned.)
void per_call_spent_veto(bool c) {
  bool ok = false, ok2 = false;
  if (c) {
    ok = mu.TryLock();
    if (!ok)
      return;
  } else {
    ok2 = mu.TryLock();
    if (ok2)
      mu.Unlock();
  }
  if (ok) {
    a = 1;
    mu.Unlock();
  }
}

// A result the loop checks and keeps across the iteration's own join is a
// possible hold on every exit of the loop: nothing after the loop checks
// it here, so it is reported at the end of the function. (The loop join
// itself is silent: the result is checked inside the loop.)
void loop_carried_result_leaks() {
  bool ok = false;
  while (cond) {
    ok = mu.TryLock(); // expected-note {{mutex acquired here}}
    if (ok)
      a = 1;
  }
} // expected-warning {{unchecked result of try-acquire; mutex 'mu' may still be held at the end of function}}

// The same loop released after it: the carried result resolves the
// post-loop branch, and the release is matched. Clean at the parent too
// -- there was no possible hold to resolve there -- so this pins that the
// hold the loop now carries out is one a later branch can still resolve.
void loop_carried_result_released_after() {
  bool ok = false;
  while (cond) {
    ok = mu.TryLock();
    if (ok)
      a = 1;
  }
  if (ok)
    mu.Unlock();
}

// The possible hold leaves through any exit edge, a break included, and
// resolves there the same way.
void loop_carried_result_break() {
  bool ok = false;
  while (cond) {
    ok = mu.TryLock();
    if (ok) {
      a = 1;
      break;
    }
  }
  if (ok)
    mu.Unlock();
}

// Two calls of one acquisition, merged by the loop condition: the spin's
// pre-loop call and its latch call. Only one of the two may be carried
// out of the loop -- the variable holds one result and the branch after
// the loop resolves it -- so a block that already records either call's
// acquisition does not take the other's. Here the loop's exit edge is an
// ordinary break, which is not the block that branches on the merge, so
// keying the fold on the patched block's own terminator found no twin and
// promoted the one acquisition twice.
void loop_carried_twin_break() {
  bool ok = mu.TryLock();
  while (!ok) {
    if (cond)
      break;
    ok = mu.TryLock();
  }
  if (ok) {
    a = 1;
    mu.Unlock();
  }
}

// The same pair with the first call already checked and released before
// the spin: its record at the head is resolved, not unresolved, and the
// fold has to see it all the same. Taking the latch's possible hold on
// top of it covers the path on which the loop never runs, and silences
// both the race and the double unlock the zero-iteration path has.
void loop_carried_twin_spent() {
  bool ok = mu.TryLock();
  if (ok) {
    a = 1;
    mu.Unlock(); // expected-note {{mutex released here}}
  }
  while (!ok)
    ok = mu.TryLock();
  a = 2;       // expected-warning {{writing variable 'a' requires holding mutex 'mu' exclusively}}
  mu.Unlock(); // expected-warning {{releasing mutex 'mu' that was not held}}
}
