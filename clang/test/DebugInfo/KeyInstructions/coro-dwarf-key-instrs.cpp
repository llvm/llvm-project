// RUN: %clang_cc1 -disable-llvm-optzns -std=c++20 \
// RUN:            -triple=x86_64 -dwarf-version=4 -debug-info-kind=limited \
// RUN:            -emit-llvm -o - %s -gkey-instructions | \
// RUN:            FileCheck %s

// Check that for the coroutine below, we mark the created DISubprogram as
// not having key instructions. This will prevent AsmPrinter from trying to
// instrument the linetable with key-instructions for source-locations in
// the coroutine scope.
//
// This is a temporary workaround for key instructions: we can instrument
// coroutine code in the future, but it hasn't been done yet.
//
// File contents copied from coro-dwarf.cpp.

namespace std {
template <typename... T> struct coroutine_traits;

template <class Promise = void> struct coroutine_handle {
  coroutine_handle() = default;
  static coroutine_handle from_address(void *) noexcept;
};
template <> struct coroutine_handle<void> {
  static coroutine_handle from_address(void *) noexcept;
  coroutine_handle() = default;
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept;
};
} // namespace std

struct suspend_always {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

template <typename... Args> struct std::coroutine_traits<void, Args...> {
  struct promise_type {
    void get_return_object() noexcept;
    suspend_always initial_suspend() noexcept;
    suspend_always final_suspend() noexcept;
    void return_void() noexcept;
    promise_type();
    ~promise_type() noexcept;
    void unhandled_exception() noexcept;
  };
};

// TODO: Not supported yet
struct CopyOnly {
  int val;
  CopyOnly(const CopyOnly &) noexcept;
  CopyOnly(CopyOnly &&) = delete;
  ~CopyOnly();
};

struct MoveOnly {
  int val;
  MoveOnly(const MoveOnly &) = delete;
  MoveOnly(MoveOnly &&) noexcept;
  ~MoveOnly();
};

struct MoveAndCopy {
  int val;
  MoveAndCopy(const MoveAndCopy &) noexcept;
  MoveAndCopy(MoveAndCopy &&) noexcept;
  ~MoveAndCopy();
};

void consume(int, int, int) noexcept;

void f_coro(int val, MoveOnly moParam, MoveAndCopy mcParam) {
  consume(val, moParam.val, mcParam.val);
  co_return;
}

// CHECK: ![[SP:[0-9]+]] = distinct !DISubprogram(name: "f_coro", linkageName: "_Z6f_coroi8MoveOnly11MoveAndCopy"
// CHECK-NOT: keyInstructions:
// CHECK: !DIFil

