// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-scoped-lock %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-scoped-lock.WarnOnUsingAndTypedef: false}}" \
// RUN:   -- -isystem %clang_tidy_headers -fno-delayed-template-parsing

#include <mutex>

template <typename T>
using Lock = std::lock_guard<T>;

using LockM = std::lock_guard<std::mutex>;

typedef std::lock_guard<std::mutex> LockDef;

void PositiveUsingDecl() {
  using std::lock_guard;

  using LockMFun = std::lock_guard<std::mutex>;
  
  typedef std::lock_guard<std::mutex> LockDefFun;
}

template <typename T>
void PositiveUsingDeclTemplate() {
  using std::lock_guard;

  using LockFunT = std::lock_guard<T>;

  using LockMFunT = std::lock_guard<std::mutex>;

  typedef std::lock_guard<std::mutex> LockDefFunT;
}
