//===-- asan_errors.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// ASan-private header for error structures.
//===----------------------------------------------------------------------===//
#ifndef ASAN_ERRORS_H
#define ASAN_ERRORS_H

#include "asan_descriptions.h"
#include "asan_scariness_score.h"

namespace __asan {

struct ErrorBase {
  ScarinessScore scariness;
};

struct ErrorStackOverflow : ErrorBase {
  u32 tid;
  uptr addr, pc, bp, sp;
  // ErrorStackOverflow never owns the context.
  void *context;
  ErrorStackOverflow() = default;
  ErrorStackOverflow(const SignalContext &sig, u32 tid_)
      : tid(tid_),
        addr(sig.addr),
        pc(sig.pc),
        bp(sig.bp),
        sp(sig.sp),
        context(sig.context) {
    scariness.Scare(10, "stack-overflow");
  }
  void Print();
};

enum ErrorKind {
  kErrorKindInvalid = 0,
  kErrorKindStackOverflow,
};

struct ErrorDescription {
  ErrorKind kind;
  union {
    ErrorStackOverflow stack_overflow;
  };
  ErrorDescription() { internal_memset(this, 0, sizeof(*this)); }
  ErrorDescription(const ErrorStackOverflow &e)  // NOLINT
      : kind(kErrorKindStackOverflow),
        stack_overflow(e) {}

  bool IsValid() { return kind != kErrorKindInvalid; }
  void Print() {
    switch (kind) {
      case kErrorKindStackOverflow:
        stack_overflow.Print();
        return;
      case kErrorKindInvalid:
        CHECK(0);
    }
    CHECK(0);
  }
};

}  // namespace __asan

#endif  // ASAN_ERRORS_H
