// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: target={{(aarch64|s390x|x86_64|arm64e)-.+}}
// UNSUPPORTED: target={{.*-windows.*}}

// *SAN does not like our clearly nonsense personality and handler functions
// which is the correct response for them, but alas we have to allow it for JITs
// because they tend to use a shared handler rather than having the handler
// within the function bounds.
// UNSUPPORTED: asan
// UNSUPPORTED: msan

#include <libunwind.h>
#include <stdint.h>
#include <stdio.h>
#include <unwind.h>

extern "C" void exit(int);
extern "C" void abort(void);

void unrelated_function() {}

int main(int, const char **) {
  unw_context_t context;
  unw_getcontext(&context);

  unw_cursor_t cursor;
  unw_init_local(&cursor, &context);

  void *ip = (void *)&unrelated_function;
#if defined(__PTRAUTH__) || __has_feature(ptrauth_calls)
  unw_word_t sp;
  if (unw_get_reg(&cursor, UNW_REG_SP, &sp) != UNW_ESUCCESS)
    abort();
  ip = ptrauth_auth_and_resign(ip, ptrauth_key_function_pointer, 0,
                               ptrauth_key_return_address, (void *)sp);
#endif
  int ret = unw_set_reg(&cursor, UNW_REG_IP, (unw_word_t)ip);
  if (ret != UNW_ESUCCESS)
    abort();
}
