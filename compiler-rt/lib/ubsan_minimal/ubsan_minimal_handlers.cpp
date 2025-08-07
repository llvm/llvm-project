#include "sanitizer_common/sanitizer_atomic.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef KERNEL_USE
extern "C" void ubsan_message(const char *msg);
static void message(const char *msg) { ubsan_message(msg); }
#else
static void message(const char *msg) { (void)write(2, msg, strlen(msg)); }
#endif

static const int kMaxCallerPcs = 20;
static __sanitizer::atomic_uintptr_t caller_pcs[kMaxCallerPcs];
// Number of elements in caller_pcs. A special value of kMaxCallerPcs + 1 means
// that "too many errors" has already been reported.
static __sanitizer::atomic_uint32_t caller_pcs_sz;

static char *append_str(const char *s, char *buf, const char *end) {
  for (const char *p = s; (buf < end) && (*p != '\0'); ++p, ++buf)
    *buf = *p;
  return buf;
}

static char *append_hex(uintptr_t d, char *buf, const char *end) {
  // Print the address by nibbles.
  for (unsigned shift = sizeof(uintptr_t) * 8; shift && buf < end;) {
    shift -= 4;
    unsigned nibble = (d >> shift) & 0xf;
    *(buf++) = nibble < 10 ? nibble + '0' : nibble - 10 + 'a';
  }
  return buf;
}

static void format_msg(const char *kind, uintptr_t caller,
                       const uintptr_t *address, char *buf, const char *end) {
  buf = append_str("ubsan: ", buf, end);
  buf = append_str(kind, buf, end);
  buf = append_str(" by 0x", buf, end);
  buf = append_hex(caller, buf, end);
  if (address) {
    buf = append_str(" address 0x", buf, end);
    buf = append_hex(*address, buf, end);
  }
  buf = append_str("\n", buf, end);
  if (buf == end)
    --buf; // Make sure we don't cause a buffer overflow.
  *buf = '\0';
}

SANITIZER_INTERFACE_WEAK_DEF(void, __ubsan_report_error, const char *kind,
                             uintptr_t caller, const uintptr_t *address) {
  if (caller == 0)
    return;
  while (true) {
    unsigned sz = __sanitizer::atomic_load_relaxed(&caller_pcs_sz);
    if (sz > kMaxCallerPcs)
      return; // early exit
    // when sz==kMaxCallerPcs print "too many errors", but only when cmpxchg
    // succeeds in order to not print it multiple times.
    if (sz > 0 && sz < kMaxCallerPcs) {
      uintptr_t p;
      for (unsigned i = 0; i < sz; ++i) {
        p = __sanitizer::atomic_load_relaxed(&caller_pcs[i]);
        if (p == 0)
          break; // Concurrent update.
        if (p == caller)
          return;
      }
      if (p == 0)
        continue; // FIXME: yield?
    }

    if (!__sanitizer::atomic_compare_exchange_strong(
            &caller_pcs_sz, &sz, sz + 1, __sanitizer::memory_order_seq_cst))
      continue; // Concurrent update! Try again from the start.

    if (sz == kMaxCallerPcs) {
      message("ubsan: too many errors\n");
      return;
    }
    __sanitizer::atomic_store_relaxed(&caller_pcs[sz], caller);

    char msg_buf[128];
    format_msg(kind, caller, address, msg_buf, msg_buf + sizeof(msg_buf));
    message(msg_buf);
  }
}

SANITIZER_INTERFACE_WEAK_DEF(void, __ubsan_report_error_fatal, const char *kind,
                             uintptr_t caller, const uintptr_t *address) {
  // Use another handlers, in case it's already overriden.
  __ubsan_report_error(kind, caller, address);
}

#if defined(__ANDROID__)
extern "C" __attribute__((weak)) void android_set_abort_message(const char *);
static void abort_with_message(const char *kind, uintptr_t caller) {
  char msg_buf[128];
  format_msg(kind, caller, msg_buf, msg_buf + sizeof(msg_buf));
  if (&android_set_abort_message)
    android_set_abort_message(msg_buf);
  abort();
}
#else
static void abort_with_message(const char *kind, uintptr_t caller) { abort(); }
#endif

#if SANITIZER_DEBUG
namespace __sanitizer {
// The DCHECK macro needs this symbol to be defined.
void NORETURN CheckFailed(const char *file, int, const char *cond, u64, u64) {
  message("Sanitizer CHECK failed: ");
  message(file);
  message(":?? : "); // FIXME: Show line number.
  message(cond);
  abort();
}
} // namespace __sanitizer
#endif

#define INTERFACE extern "C" __attribute__((visibility("default")))

#define HANDLER_RECOVER(name, kind)                                            \
  INTERFACE void __ubsan_handle_##name##_minimal() {                           \
    __ubsan_report_error(kind, GET_CALLER_PC(), nullptr);                      \
  }

#define HANDLER_NORECOVER(name, kind)                                          \
  INTERFACE void __ubsan_handle_##name##_minimal_abort() {                     \
    uintptr_t caller = GET_CALLER_PC();                                        \
    __ubsan_report_error_fatal(kind, caller, nullptr);                         \
    abort_with_message(kind, caller);                                          \
  }

#define HANDLER(name, kind)                                                    \
  HANDLER_RECOVER(name, kind)                                                  \
  HANDLER_NORECOVER(name, kind)

#define HANDLER_RECOVER_PTR(name, kind)                                        \
  INTERFACE void __ubsan_handle_##name##_minimal(const uintptr_t address) {    \
    __ubsan_report_error(kind, GET_CALLER_PC(), &address);                     \
  }

#define HANDLER_NORECOVER_PTR(name, kind)                                      \
  INTERFACE void __ubsan_handle_##name##_minimal_abort(                        \
      const uintptr_t address) {                                               \
    uintptr_t caller = GET_CALLER_PC();                                        \
    __ubsan_report_error_fatal(kind, caller, &address);                        \
    abort_with_message(kind, caller);                                          \
  }

// A version of a handler that takes a pointer to a value.
#define HANDLER_PTR(name, kind)                                                \
  HANDLER_RECOVER_PTR(name, kind)                                              \
  HANDLER_NORECOVER_PTR(name, kind)

HANDLER_PTR(type_mismatch, "type-mismatch")
HANDLER(alignment_assumption, "alignment-assumption")
HANDLER(add_overflow, "add-overflow")
HANDLER(sub_overflow, "sub-overflow")
HANDLER(mul_overflow, "mul-overflow")
HANDLER(negate_overflow, "negate-overflow")
HANDLER(divrem_overflow, "divrem-overflow")
HANDLER(shift_out_of_bounds, "shift-out-of-bounds")
HANDLER(out_of_bounds, "out-of-bounds")
HANDLER(local_out_of_bounds, "local-out-of-bounds")
HANDLER_RECOVER(builtin_unreachable, "builtin-unreachable")
HANDLER_RECOVER(missing_return, "missing-return")
HANDLER(vla_bound_not_positive, "vla-bound-not-positive")
HANDLER(float_cast_overflow, "float-cast-overflow")
HANDLER(load_invalid_value, "load-invalid-value")
HANDLER(invalid_builtin, "invalid-builtin")
HANDLER(invalid_objc_cast, "invalid-objc-cast")
HANDLER(function_type_mismatch, "function-type-mismatch")
HANDLER(implicit_conversion, "implicit-conversion")
HANDLER(nonnull_arg, "nonnull-arg")
HANDLER(nonnull_return, "nonnull-return")
HANDLER(nullability_arg, "nullability-arg")
HANDLER(nullability_return, "nullability-return")
HANDLER(pointer_overflow, "pointer-overflow")
HANDLER(cfi_check_fail, "cfi-check-fail")
