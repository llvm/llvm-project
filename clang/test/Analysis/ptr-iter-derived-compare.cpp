// RUN: %clang_analyze_cc1 -std=c++11 \
// RUN:   -analyzer-checker=core,cplusplus.NewDelete \
// RUN:   -verify %s
// expected-no-diagnostics

// Test that the analyzer correctly determines that a pointer derived through
// a pointer field (like linked list traversal) cannot equal its original value.
// This prevents false positives in patterns like STAILQ_REMOVE.

// Matches the structure in sys/queue.h STAILQ implementation
struct Foo {
  int n;
  struct { struct Foo *stqe_next; } next;
};

struct FooHead { struct Foo *stqh_first; struct Foo **stqh_last; }
foos = { nullptr, &(foos).stqh_first };

// This pattern is from STAILQ_FOREACH + STAILQ_REMOVE usage.
// Previously, the analyzer would falsely report a use-after-free because
// it could not determine that after `fi = fi->next.stqe_next`, the pointer
// `fi` cannot be equal to `foos.stqh_first`.
void test_stailq_foreach_remove() {
  bool removed;
  do {
    removed = false;
    for (Foo *fi = foos.stqh_first; fi; fi = fi->next.stqe_next) {
      if (fi->n == 1) {
        // STAILQ_REMOVE: if fi == head, remove head; else search and remove
        if (foos.stqh_first == fi) {
          foos.stqh_first = foos.stqh_first->next.stqe_next;
          if (foos.stqh_first == nullptr)
            foos.stqh_last = &foos.stqh_first;
        } else {
          Foo *curelm = foos.stqh_first;
          while (curelm->next.stqe_next != fi)
            curelm = curelm->next.stqe_next;
          if ((curelm->next.stqe_next = curelm->next.stqe_next->next.stqe_next) == nullptr)
            foos.stqh_last = &curelm->next.stqe_next;
        }
        delete fi;
        removed = true;
        break;
      }
    }
  } while (removed);
}
