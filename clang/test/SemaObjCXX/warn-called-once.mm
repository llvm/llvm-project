// RUN: %clang_cc1 -verify -fsyntax-only -fblocks -Wcompletion-handler %s

// expected-no-diagnostics

class HasCtor {
  HasCtor(void *) {}
};

void double_call_one_block(void (^completionHandler)(void)) {
  completionHandler();
  completionHandler();
  // no-warning - we don't support C++/Obj-C++ yet
}
