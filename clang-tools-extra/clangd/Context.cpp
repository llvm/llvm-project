//===--- Context.cpp -----------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "Context.h"
#include "llvm/Config/config.h"
#include <cassert>

// The thread-local Context is scoped in a function to avoid init-order issues.
// It's created by currentContext() when first needed.

#ifdef HAVE_PTHREAD_GETSPECIFIC
// We'd love to use thread_local everywhere.
// It requires support from the runtime: __cxa_thread_atexit.
// Rather than detect this, we use the pthread API where available.
#include <pthread.h>
static clang::clangd::Context &currentContext() {
  using clang::clangd::Context;
  static pthread_key_t CtxKey;

  // Once (across threads), set up pthread TLS for Context, and its destructor.
  static int Dummy = [] { // Create key only once, for all threads.
    if (auto Err = pthread_key_create(&CtxKey, /*destructor=*/+[](void *Ctx) {
          delete reinterpret_cast<Context *>(Ctx);
        }))
      llvm_unreachable(strerror(Err));
    return 0;
  }();
  (void)Dummy;

  // Now grab the current context from TLS, and create it if it doesn't exist.
  void *Ctx = pthread_getspecific(CtxKey);
  if (!Ctx) {
    Ctx = new Context();
    if (auto Err = pthread_setspecific(CtxKey, Ctx))
      llvm_unreachable(strerror(Err));
  }
  return *reinterpret_cast<Context *>(Ctx);
}
#else
// Only supported platform without pthread is windows, and thread_local works.
static clang::clangd::Context &currentContext() {
  static thread_local auto C = clang::clangd::Context::empty();
  return C;
}
#endif

namespace clang {
namespace clangd {

Context Context::empty() { return Context(/*Data=*/nullptr); }

Context::Context(std::shared_ptr<const Data> DataPtr)
    : DataPtr(std::move(DataPtr)) {}

Context Context::clone() const { return Context(DataPtr); }

const Context &Context::current() { return currentContext(); }

Context Context::swapCurrent(Context Replacement) {
  std::swap(Replacement, currentContext());
  return Replacement;
}

} // namespace clangd
} // namespace clang
