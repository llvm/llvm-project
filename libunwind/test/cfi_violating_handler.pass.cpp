// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

extern "C" int puts(const char*);
extern "C" void exit(int);
int randomUnrelatedFunction(){
  puts("Successfully dispatched to handler unrelated to actual function\n");
  exit(0);
  return 0;
};

extern "C" void _Unwind_SetIP(const void*, const void*);
extern "C" int __gxx_personality_v0(int,
                                     int actions,
                                     __SIZE_TYPE__ ,
                                     void* , void* context) {
    if (actions & 1) // Search
        return 6; // _URC_HANDLER_FOUND

    // Assign to a random unrelated function
    _Unwind_SetIP(context, (const void*)&randomUnrelatedFunction);
    return 7; // _URC_INSTALL_CONTEXT;
}

__attribute__((noinline)) extern "C" void throwAThing() {
  throw 1;
}

int main(int, const char **) {
  try {
    throwAThing();
  } catch (int) {
    puts("Called the wrong handler\n");
    return 1;
  }
  puts("Somehow returned\n");
  return 1;
}
