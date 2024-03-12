// RUN: %clang_cc1 -disable-llvm-optzns -std=c++20 \
// RUN:            -triple=x86_64 -dwarf-version=4 -debug-info-kind=limited \
// RUN:            -emit-llvm -o - %s | \
// RUN:            FileCheck %s --implicit-check-not=DILocalVariable

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
// CHECK: !{{[0-9]+}} = !DILocalVariable(name: "val", arg: 1, scope: ![[SP]], file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: !{{[0-9]+}} = !DILocalVariable(name: "moParam", arg: 2, scope: ![[SP]], file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: !{{[0-9]+}} = !DILocalVariable(name: "mcParam", arg: 3, scope: ![[SP]], file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: !{{[0-9]+}} = !DILocalVariable(name: "__promise",

// CHECK: !{{[0-9]+}} = distinct !DISubprogram(linkageName: "__await_suspend_wrapper__Z6f_coroi8MoveOnly11MoveAndCopy_init"
// CHECK-NEXT: !{{[0-9]+}} = !DIFile
// CHECK-NEXT: !{{[0-9]+}} = !DISubroutineType
// CHECK-NEXT: !{{[0-9]+}} = !DILocalVariable(arg: 1,
// CHECK-NEXT: !{{[0-9]+}} = !DILocation
// CHECK-NEXT: !{{[0-9]+}} = !DILocalVariable(arg: 2,

// CHECK: !{{[0-9]+}} = distinct !DISubprogram(linkageName: "__await_suspend_wrapper__Z6f_coroi8MoveOnly11MoveAndCopy_final"
// CHECK-NEXT: !{{[0-9]+}} = !DILocalVariable(arg: 1,
// CHECK-NEXT: !{{[0-9]+}} = !DILocation
// CHECK-NEXT: !{{[0-9]+}} = !DILocalVariable(arg: 2,
