#include <atomic>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void message(const char *msg) {
  write(2, msg, strlen(msg));
}

static const int kMaxCallerPcs = 20;
static std::atomic<void *> caller_pcs[kMaxCallerPcs];
// Number of elements in caller_pcs. A special value of kMaxCallerPcs + 1 means
// that "too many errors" has already been reported.
static std::atomic<int> caller_pcs_sz;

__attribute__((noinline))
static bool report_this_error(void *caller) {
  if (caller == nullptr) return false;
  while (true) {
    int sz = caller_pcs_sz.load(std::memory_order_relaxed);
    if (sz > kMaxCallerPcs) return false; // early exit
    // when sz==kMaxCallerPcs print "too many errors", but only when cmpxchg
    // succeeds in order to not print it multiple times.
    if (sz > 0 && sz < kMaxCallerPcs) {
      void *p;
      for (int i = 0; i < sz; ++i) {
        p = caller_pcs[i].load(std::memory_order_relaxed);
        if (p == nullptr) break; // Concurrent update.
        if (p == caller) return false;
      }
      if (p == nullptr) continue; // FIXME: yield?
    }

    if (!caller_pcs_sz.compare_exchange_strong(sz, sz + 1))
      continue; // Concurrent update! Try again from the start.

    if (sz == kMaxCallerPcs) {
      message("ubsan: too many errors\n");
      return false;
    }
    caller_pcs[sz].store(caller, std::memory_order_relaxed);
    return true;
  }
}

#if defined(__ANDROID__)
extern "C" __attribute__((weak)) void android_set_abort_message(const char *);
static void abort_with_message(const char *msg) {
  if (&android_set_abort_message) android_set_abort_message(msg);
  abort();
}
#else
static void abort_with_message(const char *) { abort(); }
#endif

#define INTERFACE extern "C" __attribute__((visibility("default")))

// FIXME: add caller pc to the error message (possibly as "ubsan: error-type
// @1234ABCD").
#define HANDLER_RECOVER(name, msg)                               \
  INTERFACE void __ubsan_handle_##name##_minimal() {             \
    if (!report_this_error(__builtin_return_address(0))) return; \
    message("ubsan: " msg "\n");                                 \
  }

#define HANDLER_NORECOVER(name, msg)                             \
  INTERFACE void __ubsan_handle_##name##_minimal_abort() {       \
    message("ubsan: " msg "\n");                                 \
    abort_with_message("ubsan: " msg);                           \
  }

#define HANDLER(name, msg)                                       \
  HANDLER_RECOVER(name, msg)                                     \
  HANDLER_NORECOVER(name, msg)

HANDLER(type_mismatch, "type-mismatch")
HANDLER(add_overflow, "add-overflow")
HANDLER(sub_overflow, "sub-overflow")
HANDLER(mul_overflow, "mul-overflow")
HANDLER(negate_overflow, "negate-overflow")
HANDLER(divrem_overflow, "divrem-overflow")
HANDLER(shift_out_of_bounds, "shift-out-of-bounds")
HANDLER(out_of_bounds, "out-of-bounds")
HANDLER_RECOVER(builtin_unreachable, "builtin-unreachable")
HANDLER_RECOVER(missing_return, "missing-return")
HANDLER(vla_bound_not_positive, "vla-bound-not-positive")
HANDLER(float_cast_overflow, "float-cast-overflow")
HANDLER(load_invalid_value, "load-invalid-value")
HANDLER(invalid_builtin, "invalid-builtin")
HANDLER(function_type_mismatch, "function-type-mismatch")
HANDLER(nonnull_arg, "nonnull-arg")
HANDLER(nonnull_return, "nonnull-return")
HANDLER(nullability_arg, "nullability-arg")
HANDLER(nullability_return, "nullability-return")
HANDLER(pointer_overflow, "pointer-overflow")
HANDLER(cfi_check_fail, "cfi-check-fail")
