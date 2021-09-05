// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: linux

// Basic test for _Unwind_ForcedUnwind.
// See libcxxabi/test/forced_unwind* tests too.

#include <assert.h>
#include <dlfcn.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <unwind.h>

void foo();
_Unwind_Exception ex;

_Unwind_Reason_Code stop(int version, _Unwind_Action actions,
                         _Unwind_Exception_Class exceptionClass,
                         _Unwind_Exception *exceptionObject,
                         struct _Unwind_Context *context,
                         void *stop_parameter) {
  assert(version == 1);
  assert((actions & _UA_FORCE_UNWIND) != 0);
  (void)exceptionClass;
  assert(exceptionObject == &ex);
  assert(stop_parameter == &foo);

  Dl_info info = {0, 0, 0, 0};

  // Unwind util the main is reached, above frames depend on the platform and
  // architecture.
  if (dladdr(reinterpret_cast<void *>(_Unwind_GetIP(context)), &info) &&
      info.dli_sname && !strcmp("main", info.dli_sname)) {
    _Exit(0);
  }
  return _URC_NO_REASON;
}

__attribute__((noinline)) void foo() {

  // Arm EHABI defines struct _Unwind_Control_Block as exception
  // object. Ensure struct _Unwind_Exception* work there too,
  // because _Unwind_Exception in this case is just an alias.
  struct _Unwind_Exception *e = &ex;
#if defined(_LIBUNWIND_ARM_EHABI)
  // Create a mock exception object.
  memset(e, '\0', sizeof(*e));
  strcpy(reinterpret_cast<char *>(&e->exception_class), "CLNGUNW");
#endif
  _Unwind_ForcedUnwind(e, stop, (void *)&foo);
}

int main() {
  foo();
  return -2;
}
