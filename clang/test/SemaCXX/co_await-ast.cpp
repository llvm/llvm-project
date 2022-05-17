// RUN: %clang_cc1 -std=c++20 -fsyntax-only -ast-dump -ast-dump-filter=foo %s | FileCheck %s --strict-whitespace

namespace std {
template <typename, typename...> struct coroutine_traits;
template <typename> struct coroutine_handle {
  template <typename U>
  coroutine_handle(coroutine_handle<U> &&) noexcept;
  static coroutine_handle from_address(void *__addr) noexcept;
};
} // namespace std

struct executor {};
struct awaitable {};
struct awaitable_frame {
  awaitable get_return_object();
  void return_void();
  void unhandled_exception();
  struct result_t {
    ~result_t();
    bool await_ready() const noexcept;
    void await_suspend(std::coroutine_handle<void>) noexcept;
    void await_resume() const noexcept;
  };
  result_t initial_suspend() noexcept;
  result_t final_suspend() noexcept;
  result_t await_transform(executor) noexcept;
};

namespace std {
template <>
struct coroutine_traits<awaitable> {
  typedef awaitable_frame promise_type;
};
} // namespace std

awaitable foo() {
  co_await executor();
}

// Check that CoawaitExpr contains the correct subexpressions, including
// the operand expression as written in the source.

// CHECK-LABEL: Dumping foo:
// CHECK: FunctionDecl {{.*}} foo 'awaitable ()'
// CHECK: `-CoroutineBodyStmt {{.*}}
// CHECK:   |-CompoundStmt {{.*}}
// CHECK:   | `-ExprWithCleanups {{.*}} 'void'
// CHECK:   |   `-CoawaitExpr {{.*}} 'void'
// CHECK:   |     |-CXXTemporaryObjectExpr {{.*}} 'executor' 'void () noexcept' zeroing
// CHECK:   |     |-MaterializeTemporaryExpr {{.*}} 'awaitable_frame::result_t' lvalue
// CHECK:   |     | `-CXXBindTemporaryExpr {{.*}} 'awaitable_frame::result_t' (CXXTemporary {{.*}})
// CHECK:   |     |   `-CXXMemberCallExpr {{.*}} 'awaitable_frame::result_t'
// CHECK:   |     |     |-MemberExpr {{.*}} '<bound member function type>' .await_transform {{.*}}
// CHECK:   |     |     | `-DeclRefExpr {{.*}} 'std::coroutine_traits<awaitable>::promise_type':'awaitable_frame' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<awaitable>::promise_type':'awaitable_frame'
// CHECK:   |     |     `-CXXTemporaryObjectExpr {{.*}} 'executor' 'void () noexcept' zeroing
// CHECK:   |     |-ExprWithCleanups {{.*}} 'bool'
// CHECK:   |     | `-CXXMemberCallExpr {{.*}} 'bool'
// CHECK:   |     |   `-MemberExpr {{.*}} '<bound member function type>' .await_ready {{.*}}
// CHECK:   |     |     `-ImplicitCastExpr {{.*}} 'const awaitable_frame::result_t' lvalue <NoOp>
// CHECK:   |     |       `-OpaqueValueExpr {{.*}} 'awaitable_frame::result_t' lvalue
// CHECK:   |     |         `-MaterializeTemporaryExpr {{.*}} 'awaitable_frame::result_t' lvalue
// CHECK:   |     |           `-CXXBindTemporaryExpr {{.*}} 'awaitable_frame::result_t' (CXXTemporary {{.*}})
// CHECK:   |     |             `-CXXMemberCallExpr {{.*}} 'awaitable_frame::result_t'
// CHECK:   |     |               |-MemberExpr {{.*}} '<bound member function type>' .await_transform {{.*}}
// CHECK:   |     |               | `-DeclRefExpr {{.*}} 'std::coroutine_traits<awaitable>::promise_type':'awaitable_frame' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<awaitable>::promise_type':'awaitable_frame'
// CHECK:   |     |               `-CXXTemporaryObjectExpr {{.*}} 'executor' 'void () noexcept' zeroing
// CHECK:   |     |-ExprWithCleanups {{.*}} 'void'
// CHECK:   |     | `-CXXMemberCallExpr {{.*}} 'void'
// CHECK:   |     |   |-MemberExpr {{.*}} '<bound member function type>' .await_suspend {{.*}}
// CHECK:   |     |   | `-OpaqueValueExpr {{.*}} 'awaitable_frame::result_t' lvalue
// CHECK:   |     |   |   `-MaterializeTemporaryExpr {{.*}} 'awaitable_frame::result_t' lvalue
// CHECK:   |     |   |     `-CXXBindTemporaryExpr {{.*}} 'awaitable_frame::result_t' (CXXTemporary {{.*}})
// CHECK:   |     |   |       `-CXXMemberCallExpr {{.*}} 'awaitable_frame::result_t'
// CHECK:   |     |   |         |-MemberExpr {{.*}} '<bound member function type>' .await_transform {{.*}}
// CHECK:   |     |   |         | `-DeclRefExpr {{.*}} 'std::coroutine_traits<awaitable>::promise_type':'awaitable_frame' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<awaitable>::promise_type':'awaitable_frame'
// CHECK:   |     |   |         `-CXXTemporaryObjectExpr {{.*}} 'executor' 'void () noexcept' zeroing
// CHECK:   |     |   `-ImplicitCastExpr {{.*}} 'std::coroutine_handle<void>':'std::coroutine_handle<void>' <ConstructorConversion>
// CHECK:   |     |     `-CXXConstructExpr {{.*}} 'std::coroutine_handle<void>':'std::coroutine_handle<void>' 'void (coroutine_handle<awaitable_frame> &&) noexcept'
// CHECK:   |     |       `-MaterializeTemporaryExpr {{.*}} 'std::coroutine_handle<awaitable_frame>' xvalue
// CHECK:   |     |         `-CallExpr {{.*}} 'std::coroutine_handle<awaitable_frame>'
// CHECK:   |     |           |-ImplicitCastExpr {{.*}} 'std::coroutine_handle<awaitable_frame> (*)(void *) noexcept' <FunctionToPointerDecay>
// CHECK:   |     |           | `-DeclRefExpr {{.*}} 'std::coroutine_handle<awaitable_frame> (void *) noexcept' lvalue CXXMethod {{.*}} 'from_address' 'std::coroutine_handle<awaitable_frame> (void *) noexcept'
// CHECK:   |     |           `-CallExpr {{.*}} 'void *'
// CHECK:   |     |             `-ImplicitCastExpr {{.*}} 'void *(*)() noexcept' <FunctionToPointerDecay>
// CHECK:   |     |               `-DeclRefExpr {{.*}} 'void *() noexcept' lvalue Function {{.*}} '__builtin_coro_frame' 'void *() noexcept'
// CHECK:   |     `-CXXMemberCallExpr {{.*}} 'void'
// CHECK:   |       `-MemberExpr {{.*}} '<bound member function type>' .await_resume {{.*}}
// CHECK:   |         `-ImplicitCastExpr {{.*}} 'const awaitable_frame::result_t' lvalue <NoOp>
// CHECK:   |           `-OpaqueValueExpr {{.*}} 'awaitable_frame::result_t' lvalue
// CHECK:   |             `-MaterializeTemporaryExpr {{.*}} 'awaitable_frame::result_t' lvalue
// CHECK:   |               `-CXXBindTemporaryExpr {{.*}} 'awaitable_frame::result_t' (CXXTemporary {{.*}})
// CHECK:   |                 `-CXXMemberCallExpr {{.*}} 'awaitable_frame::result_t'
// CHECK:   |                   |-MemberExpr {{.*}} '<bound member function type>' .await_transform {{.*}}
// CHECK:   |                   | `-DeclRefExpr {{.*}} 'std::coroutine_traits<awaitable>::promise_type':'awaitable_frame' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<awaitable>::promise_type':'awaitable_frame'
// CHECK:   |                   `-CXXTemporaryObjectExpr {{.*}} <col:12, col:21> 'executor' 'void () noexcept' zeroing

// Rest of the generated coroutine statements omitted.
