// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -std=c++20 \
// RUN: -ast-dump -ast-dump-filter test | FileCheck %s

#include "Inputs/std-coroutine.h"

using namespace std;

struct Task {
 struct promise_type {
 std::suspend_always initial_suspend() { return {}; }
 Task get_return_object() {
 return std::coroutine_handle<promise_type>::from_promise(*this);
 }
 std::suspend_always final_suspend() noexcept { return {}; }
 std::suspend_always return_void() { return {}; }
 void unhandled_exception() {}

 auto await_transform(int s) {
 struct awaiter {
 promise_type *promise;
 bool await_ready() { return true; }
 int await_resume() { return 1; }
 void await_suspend(std::coroutine_handle<>) {}
 };

 return awaiter{this};
 }
 };

 Task(std::coroutine_handle<promise_type> promise);

 std::coroutine_handle<promise_type> handle;
};

Task test() {
 co_await 1;
}

// CHECK: FunctionDecl {{.*}} test 'Task ()'
// CHECK-NEXT: `-CoroutineBodyStmt {{.*}} 
// CHECK-NEXT:   |-CompoundStmt {{.*}} 
// CHECK-NEXT:   | `-ExprWithCleanups {{.*}} 'int'
// CHECK-NEXT:   |   `-CoawaitExpr {{.*}} 'int'
// CHECK-NEXT:   |     |-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:   |     |-MaterializeTemporaryExpr {{.*}} 'awaiter' lvalue
// CHECK-NEXT:   |     | `-CXXMemberCallExpr {{.*}} 'awaiter'
// CHECK-NEXT:   |     |   |-MemberExpr {{.*}} '<bound member function type>' .await_transform {{.*}}
// CHECK-NEXT:   |     |   | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |     |   `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:   |     |-ExprWithCleanups {{.*}} 'bool'
// CHECK-NEXT:   |     | `-CXXMemberCallExpr {{.*}} 'bool'
// CHECK-NEXT:   |     |   `-MemberExpr {{.*}} '<bound member function type>' .await_ready {{.*}}
// CHECK-NEXT:   |     |     `-OpaqueValueExpr {{.*}} 'awaiter' lvalue
// CHECK-NEXT:   |     |       `-MaterializeTemporaryExpr {{.*}} 'awaiter' lvalue
// CHECK-NEXT:   |     |         `-CXXMemberCallExpr {{.*}} 'awaiter'
// CHECK-NEXT:   |     |           |-MemberExpr {{.*}} '<bound member function type>' .await_transform {{.*}}
// CHECK-NEXT:   |     |           | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |     |           `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:   |     |-ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT:   |     | `-CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT:   |     |   |-MemberExpr {{.*}} '<bound member function type>' .await_suspend {{.*}}
// CHECK-NEXT:   |     |   | `-OpaqueValueExpr {{.*}} 'awaiter' lvalue
// CHECK-NEXT:   |     |   |   `-MaterializeTemporaryExpr {{.*}} 'awaiter' lvalue
// CHECK-NEXT:   |     |   |     `-CXXMemberCallExpr {{.*}} 'awaiter'
// CHECK-NEXT:   |     |   |       |-MemberExpr {{.*}} '<bound member function type>' .await_transform {{.*}}
// CHECK-NEXT:   |     |   |       | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |     |   |       `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:   |     |   `-CXXConstructExpr {{.*}} 'std::coroutine_handle<>':'std::coroutine_handle<void>' 'void (coroutine_handle<void> &&) noexcept'
// CHECK-NEXT:   |     |     `-ImplicitCastExpr {{.*}} 'coroutine_handle<void>':'std::coroutine_handle<void>' xvalue <DerivedToBase (coroutine_handle)>
// CHECK-NEXT:   |     |       `-MaterializeTemporaryExpr {{.*}} 'coroutine_handle<Task::promise_type>':'std::coroutine_handle<Task::promise_type>' xvalue
// CHECK-NEXT:   |     |         `-CallExpr {{.*}} 'coroutine_handle<Task::promise_type>':'std::coroutine_handle<Task::promise_type>'
// CHECK-NEXT:   |     |           |-ImplicitCastExpr {{.*}} 'coroutine_handle<Task::promise_type> (*)(void *) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   |     |           | `-DeclRefExpr {{.*}} 'coroutine_handle<Task::promise_type> (void *) noexcept' lvalue CXXMethod {{.*}} 'from_address' 'coroutine_handle<Task::promise_type> (void *) noexcept'
// CHECK-NEXT:   |     |           `-CallExpr {{.*}} 'void *'
// CHECK-NEXT:   |     |             `-ImplicitCastExpr {{.*}} 'void *(*)() noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   |     |               `-DeclRefExpr {{.*}} 'void *() noexcept' lvalue Function {{.*}} '__builtin_coro_frame' 'void *() noexcept'
// CHECK-NEXT:   |     `-CXXMemberCallExpr {{.*}} 'int'
// CHECK-NEXT:   |       `-MemberExpr {{.*}} '<bound member function type>' .await_resume {{.*}}
// CHECK-NEXT:   |         `-OpaqueValueExpr {{.*}} 'awaiter' lvalue
// CHECK-NEXT:   |           `-MaterializeTemporaryExpr {{.*}} 'awaiter' lvalue
// CHECK-NEXT:   |             `-CXXMemberCallExpr {{.*}} 'awaiter'
// CHECK-NEXT:   |               |-MemberExpr {{.*}} '<bound member function type>' .await_transform {{.*}}
// CHECK-NEXT:   |               | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |               `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:   |-DeclStmt {{.*}} 
// CHECK-NEXT:   | `-VarDecl {{.*}} implicit used __promise 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' callinit
// CHECK-NEXT:   |   |-CXXConstructExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' 'void () noexcept'
// CHECK-NEXT:   |   `-typeDetails: TypedefType {{.*}} 'std::coroutine_traits<Task>::promise_type' sugar
// CHECK-NEXT:   |     |-TypeAlias {{.*}} 'promise_type'
// CHECK-NEXT:   |     `-typeDetails: ElaboratedType {{.*}} 'typename Task::promise_type' sugar
// CHECK-NEXT:   |       `-typeDetails: RecordType {{.*}} 'Task::promise_type'
// CHECK-NEXT:   |         `-CXXRecord {{.*}} 'promise_type'
// CHECK-NEXT:   |-ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT:   | `-CoawaitExpr {{.*}} 'void' implicit
// CHECK-NEXT:   |   |-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   | `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
// CHECK-NEXT:   |   |   `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |   |-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   |   `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
// CHECK-NEXT:   |   |     `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |   |-ExprWithCleanups {{.*}} 'bool'
// CHECK-NEXT:   |   | `-CXXMemberCallExpr {{.*}} 'bool'
// CHECK-NEXT:   |   |   `-MemberExpr {{.*}} '<bound member function type>' .await_ready {{.*}}
// CHECK-NEXT:   |   |     `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   |       `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   |         `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   |           `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
// CHECK-NEXT:   |   |             `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |   |-ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT:   |   | `-CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT:   |   |   |-MemberExpr {{.*}} '<bound member function type>' .await_suspend {{.*}}
// CHECK-NEXT:   |   |   | `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   |   |   `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   |   |     `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   |   |       `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
// CHECK-NEXT:   |   |   |         `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |   |   `-CXXConstructExpr {{.*}} 'coroutine_handle<>':'std::coroutine_handle<void>' 'void (coroutine_handle<void> &&) noexcept'
// CHECK-NEXT:   |   |     `-ImplicitCastExpr {{.*}} 'coroutine_handle<void>':'std::coroutine_handle<void>' xvalue <DerivedToBase (coroutine_handle)>
// CHECK-NEXT:   |   |       `-MaterializeTemporaryExpr {{.*}} 'coroutine_handle<Task::promise_type>':'std::coroutine_handle<Task::promise_type>' xvalue
// CHECK-NEXT:   |   |         `-CallExpr {{.*}} 'coroutine_handle<Task::promise_type>':'std::coroutine_handle<Task::promise_type>'
// CHECK-NEXT:   |   |           |-ImplicitCastExpr {{.*}} 'coroutine_handle<Task::promise_type> (*)(void *) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   |   |           | `-DeclRefExpr {{.*}} 'coroutine_handle<Task::promise_type> (void *) noexcept' lvalue CXXMethod {{.*}} 'from_address' 'coroutine_handle<Task::promise_type> (void *) noexcept'
// CHECK-NEXT:   |   |           `-CallExpr {{.*}} 'void *'
// CHECK-NEXT:   |   |             `-ImplicitCastExpr {{.*}} 'void *(*)() noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   |   |               `-DeclRefExpr {{.*}} 'void *() noexcept' lvalue Function {{.*}} '__builtin_coro_frame' 'void *() noexcept'
// CHECK-NEXT:   |   `-CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT:   |     `-MemberExpr {{.*}} '<bound member function type>' .await_resume {{.*}}
// CHECK-NEXT:   |       `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |         `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |           `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |             `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
// CHECK-NEXT:   |               `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |-ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT:   | `-CoawaitExpr {{.*}} 'void' implicit
// CHECK-NEXT:   |   |-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   | `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
// CHECK-NEXT:   |   |   `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |   |-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   |   `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
// CHECK-NEXT:   |   |     `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |   |-ExprWithCleanups {{.*}} 'bool'
// CHECK-NEXT:   |   | `-CXXMemberCallExpr {{.*}} 'bool'
// CHECK-NEXT:   |   |   `-MemberExpr {{.*}} '<bound member function type>' .await_ready {{.*}}
// CHECK-NEXT:   |   |     `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   |       `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   |         `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   |           `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
// CHECK-NEXT:   |   |             `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |   |-ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT:   |   | `-CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT:   |   |   |-MemberExpr {{.*}} '<bound member function type>' .await_suspend {{.*}}
// CHECK-NEXT:   |   |   | `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   |   |   `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   |   |     `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   |   |       `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
// CHECK-NEXT:   |   |   |         `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |   |   `-CXXConstructExpr {{.*}} 'coroutine_handle<>':'std::coroutine_handle<void>' 'void (coroutine_handle<void> &&) noexcept'
// CHECK-NEXT:   |   |     `-ImplicitCastExpr {{.*}} 'coroutine_handle<void>':'std::coroutine_handle<void>' xvalue <DerivedToBase (coroutine_handle)>
// CHECK-NEXT:   |   |       `-MaterializeTemporaryExpr {{.*}} 'coroutine_handle<Task::promise_type>':'std::coroutine_handle<Task::promise_type>' xvalue
// CHECK-NEXT:   |   |         `-CallExpr {{.*}} 'coroutine_handle<Task::promise_type>':'std::coroutine_handle<Task::promise_type>'
// CHECK-NEXT:   |   |           |-ImplicitCastExpr {{.*}} 'coroutine_handle<Task::promise_type> (*)(void *) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   |   |           | `-DeclRefExpr {{.*}} 'coroutine_handle<Task::promise_type> (void *) noexcept' lvalue CXXMethod {{.*}} 'from_address' 'coroutine_handle<Task::promise_type> (void *) noexcept'
// CHECK-NEXT:   |   |           `-CallExpr {{.*}} 'void *'
// CHECK-NEXT:   |   |             `-ImplicitCastExpr {{.*}} 'void *(*)() noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   |   |               `-DeclRefExpr {{.*}} 'void *() noexcept' lvalue Function {{.*}} '__builtin_coro_frame' 'void *() noexcept'
// CHECK-NEXT:   |   `-CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT:   |     `-MemberExpr {{.*}} '<bound member function type>' .await_resume {{.*}}
// CHECK-NEXT:   |       `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |         `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |           `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |             `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
// CHECK-NEXT:   |               `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |-<<<NULL>>>
// CHECK-NEXT:   |-CoreturnStmt {{.*}} implicit
// CHECK-NEXT:   | |-<<<NULL>>>
// CHECK-NEXT:   | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   `-MemberExpr {{.*}} '<bound member function type>' .return_void {{.*}}
// CHECK-NEXT:   |     `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |-CallExpr {{.*}} 'void *'
// CHECK-NEXT:   | |-ImplicitCastExpr {{.*}} 'void *(*)(__size_t)' <FunctionToPointerDecay>
// CHECK-NEXT:   | | `-DeclRefExpr {{.*}} 'void *(__size_t)' lvalue Function {{.*}} 'operator new' 'void *(__size_t)'
// CHECK-NEXT:   | `-CallExpr {{.*}} '__size_t':'unsigned long'
// CHECK-NEXT:   |   `-ImplicitCastExpr {{.*}} '__size_t (*)() noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   |     `-DeclRefExpr {{.*}} '__size_t () noexcept' lvalue Function {{.*}} '__builtin_coro_size' '__size_t () noexcept'
// CHECK-NEXT:   |-CallExpr {{.*}} 'void'
// CHECK-NEXT:   | |-ImplicitCastExpr {{.*}} 'void (*)(void *, __size_t) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   | | `-DeclRefExpr {{.*}} 'void (void *, __size_t) noexcept' lvalue Function {{.*}} 'operator delete' 'void (void *, __size_t) noexcept'
// CHECK-NEXT:   | |-CallExpr {{.*}} 'void *'
// CHECK-NEXT:   | | |-ImplicitCastExpr {{.*}} 'void *(*)(void *) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   | | | `-DeclRefExpr {{.*}} 'void *(void *) noexcept' lvalue Function {{.*}} '__builtin_coro_free' 'void *(void *) noexcept'
// CHECK-NEXT:   | | `-CallExpr {{.*}} 'void *'
// CHECK-NEXT:   | |   `-ImplicitCastExpr {{.*}} 'void *(*)() noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   | |     `-DeclRefExpr {{.*}} 'void *() noexcept' lvalue Function {{.*}} '__builtin_coro_frame' 'void *() noexcept'
// CHECK-NEXT:   | `-CallExpr {{.*}} '__size_t':'unsigned long'
// CHECK-NEXT:   |   `-ImplicitCastExpr {{.*}} '__size_t (*)() noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   |     `-DeclRefExpr {{.*}} '__size_t () noexcept' lvalue Function {{.*}} '__builtin_coro_size' '__size_t () noexcept'
// CHECK-NEXT:   |-<<<NULL>>>
// CHECK-NEXT:   |-CXXMemberCallExpr {{.*}} 'Task'
// CHECK-NEXT:   | `-MemberExpr {{.*}} '<bound member function type>' .get_return_object {{.*}}
// CHECK-NEXT:   |   `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |-ReturnStmt {{.*}} 
// CHECK-NEXT:   | `-CXXMemberCallExpr {{.*}} 'Task'
// CHECK-NEXT:   |   `-MemberExpr {{.*}} '<bound member function type>' .get_return_object {{.*}}
// CHECK-NEXT:   |     `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   `-<<<NULL>>>

Task test2() {
 co_return;
}

// CHECK: FunctionDecl {{.*}} test2 'Task ()'
// CHECK-NEXT: `-CoroutineBodyStmt {{.*}} 
// CHECK-NEXT:   |-CompoundStmt {{.*}} 
// CHECK-NEXT:   | `-CoreturnStmt {{.*}} 
// CHECK-NEXT:   |   |-<<<NULL>>>
// CHECK-NEXT:   |   `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |     `-MemberExpr {{.*}} '<bound member function type>' .return_void {{.*}}
// CHECK-NEXT:   |       `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |-DeclStmt {{.*}} 
// CHECK-NEXT:   | `-VarDecl {{.*}} implicit used __promise 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' callinit
// CHECK-NEXT:   |   |-CXXConstructExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' 'void () noexcept'
// CHECK-NEXT:   |   `-typeDetails: TypedefType {{.*}} 'std::coroutine_traits<Task>::promise_type' sugar
// CHECK-NEXT:   |     |-TypeAlias {{.*}} 'promise_type'
// CHECK-NEXT:   |     `-typeDetails: ElaboratedType {{.*}} 'typename Task::promise_type' sugar
// CHECK-NEXT:   |       `-typeDetails: RecordType {{.*}} 'Task::promise_type'
// CHECK-NEXT:   |         `-CXXRecord {{.*}} 'promise_type'
// CHECK-NEXT:   |-ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT:   | `-CoawaitExpr {{.*}} 'void' implicit
// CHECK-NEXT:   |   |-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   | `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
// CHECK-NEXT:   |   |   `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |   |-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   |   `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
// CHECK-NEXT:   |   |     `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |   |-ExprWithCleanups {{.*}} 'bool'
// CHECK-NEXT:   |   | `-CXXMemberCallExpr {{.*}} 'bool'
// CHECK-NEXT:   |   |   `-MemberExpr {{.*}} '<bound member function type>' .await_ready {{.*}}
// CHECK-NEXT:   |   |     `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   |       `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   |         `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   |           `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
// CHECK-NEXT:   |   |             `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |   |-ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT:   |   | `-CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT:   |   |   |-MemberExpr {{.*}} '<bound member function type>' .await_suspend {{.*}}
// CHECK-NEXT:   |   |   | `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   |   |   `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   |   |     `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   |   |       `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
// CHECK-NEXT:   |   |   |         `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |   |   `-CXXConstructExpr {{.*}} 'coroutine_handle<>':'std::coroutine_handle<void>' 'void (coroutine_handle<void> &&) noexcept'
// CHECK-NEXT:   |   |     `-ImplicitCastExpr {{.*}} 'coroutine_handle<void>':'std::coroutine_handle<void>' xvalue <DerivedToBase (coroutine_handle)>
// CHECK-NEXT:   |   |       `-MaterializeTemporaryExpr {{.*}} 'coroutine_handle<Task::promise_type>':'std::coroutine_handle<Task::promise_type>' xvalue
// CHECK-NEXT:   |   |         `-CallExpr {{.*}} 'coroutine_handle<Task::promise_type>':'std::coroutine_handle<Task::promise_type>'
// CHECK-NEXT:   |   |           |-ImplicitCastExpr {{.*}} 'coroutine_handle<Task::promise_type> (*)(void *) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   |   |           | `-DeclRefExpr {{.*}} 'coroutine_handle<Task::promise_type> (void *) noexcept' lvalue CXXMethod {{.*}} 'from_address' 'coroutine_handle<Task::promise_type> (void *) noexcept'
// CHECK-NEXT:   |   |           `-CallExpr {{.*}} 'void *'
// CHECK-NEXT:   |   |             `-ImplicitCastExpr {{.*}} 'void *(*)() noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   |   |               `-DeclRefExpr {{.*}} 'void *() noexcept' lvalue Function {{.*}} '__builtin_coro_frame' 'void *() noexcept'
// CHECK-NEXT:   |   `-CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT:   |     `-MemberExpr {{.*}} '<bound member function type>' .await_resume {{.*}}
// CHECK-NEXT:   |       `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |         `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |           `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |             `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
// CHECK-NEXT:   |               `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |-ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT:   | `-CoawaitExpr {{.*}} 'void' implicit
// CHECK-NEXT:   |   |-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   | `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
// CHECK-NEXT:   |   |   `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |   |-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   |   `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
// CHECK-NEXT:   |   |     `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |   |-ExprWithCleanups {{.*}} 'bool'
// CHECK-NEXT:   |   | `-CXXMemberCallExpr {{.*}} 'bool'
// CHECK-NEXT:   |   |   `-MemberExpr {{.*}} '<bound member function type>' .await_ready {{.*}}
// CHECK-NEXT:   |   |     `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   |       `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   |         `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   |           `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
// CHECK-NEXT:   |   |             `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |   |-ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT:   |   | `-CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT:   |   |   |-MemberExpr {{.*}} '<bound member function type>' .await_suspend {{.*}}
// CHECK-NEXT:   |   |   | `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   |   |   `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |   |   |     `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   |   |       `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
// CHECK-NEXT:   |   |   |         `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |   |   `-CXXConstructExpr {{.*}} 'coroutine_handle<>':'std::coroutine_handle<void>' 'void (coroutine_handle<void> &&) noexcept'
// CHECK-NEXT:   |   |     `-ImplicitCastExpr {{.*}} 'coroutine_handle<void>':'std::coroutine_handle<void>' xvalue <DerivedToBase (coroutine_handle)>
// CHECK-NEXT:   |   |       `-MaterializeTemporaryExpr {{.*}} 'coroutine_handle<Task::promise_type>':'std::coroutine_handle<Task::promise_type>' xvalue
// CHECK-NEXT:   |   |         `-CallExpr {{.*}} 'coroutine_handle<Task::promise_type>':'std::coroutine_handle<Task::promise_type>'
// CHECK-NEXT:   |   |           |-ImplicitCastExpr {{.*}} 'coroutine_handle<Task::promise_type> (*)(void *) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   |   |           | `-DeclRefExpr {{.*}} 'coroutine_handle<Task::promise_type> (void *) noexcept' lvalue CXXMethod {{.*}} 'from_address' 'coroutine_handle<Task::promise_type> (void *) noexcept'
// CHECK-NEXT:   |   |           `-CallExpr {{.*}} 'void *'
// CHECK-NEXT:   |   |             `-ImplicitCastExpr {{.*}} 'void *(*)() noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   |   |               `-DeclRefExpr {{.*}} 'void *() noexcept' lvalue Function {{.*}} '__builtin_coro_frame' 'void *() noexcept'
// CHECK-NEXT:   |   `-CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT:   |     `-MemberExpr {{.*}} '<bound member function type>' .await_resume {{.*}}
// CHECK-NEXT:   |       `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |         `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK-NEXT:   |           `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |             `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
// CHECK-NEXT:   |               `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |-<<<NULL>>>
// CHECK-NEXT:   |-CoreturnStmt {{.*}} implicit
// CHECK-NEXT:   | |-<<<NULL>>>
// CHECK-NEXT:   | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK-NEXT:   |   `-MemberExpr {{.*}} '<bound member function type>' .return_void {{.*}}
// CHECK-NEXT:   |     `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |-CallExpr {{.*}} 'void *'
// CHECK-NEXT:   | |-ImplicitCastExpr {{.*}} 'void *(*)(__size_t)' <FunctionToPointerDecay>
// CHECK-NEXT:   | | `-DeclRefExpr {{.*}} 'void *(__size_t)' lvalue Function {{.*}} 'operator new' 'void *(__size_t)'
// CHECK-NEXT:   | `-CallExpr {{.*}} '__size_t':'unsigned long'
// CHECK-NEXT:   |   `-ImplicitCastExpr {{.*}} '__size_t (*)() noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   |     `-DeclRefExpr {{.*}} '__size_t () noexcept' lvalue Function {{.*}} '__builtin_coro_size' '__size_t () noexcept'
// CHECK-NEXT:   |-CallExpr {{.*}} 'void'
// CHECK-NEXT:   | |-ImplicitCastExpr {{.*}} 'void (*)(void *, __size_t) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   | | `-DeclRefExpr {{.*}} 'void (void *, __size_t) noexcept' lvalue Function {{.*}} 'operator delete' 'void (void *, __size_t) noexcept'
// CHECK-NEXT:   | |-CallExpr {{.*}} 'void *'
// CHECK-NEXT:   | | |-ImplicitCastExpr {{.*}} 'void *(*)(void *) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   | | | `-DeclRefExpr {{.*}} 'void *(void *) noexcept' lvalue Function {{.*}} '__builtin_coro_free' 'void *(void *) noexcept'
// CHECK-NEXT:   | | `-CallExpr {{.*}} 'void *'
// CHECK-NEXT:   | |   `-ImplicitCastExpr {{.*}} 'void *(*)() noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   | |     `-DeclRefExpr {{.*}} 'void *() noexcept' lvalue Function {{.*}} '__builtin_coro_frame' 'void *() noexcept'
// CHECK-NEXT:   | `-CallExpr {{.*}} '__size_t':'unsigned long'
// CHECK-NEXT:   |   `-ImplicitCastExpr {{.*}} '__size_t (*)() noexcept' <FunctionToPointerDecay>
// CHECK-NEXT:   |     `-DeclRefExpr {{.*}} '__size_t () noexcept' lvalue Function {{.*}} '__builtin_coro_size' '__size_t () noexcept'
// CHECK-NEXT:   |-<<<NULL>>>
// CHECK-NEXT:   |-CXXMemberCallExpr {{.*}} 'Task'
// CHECK-NEXT:   | `-MemberExpr {{.*}} '<bound member function type>' .get_return_object {{.*}}
// CHECK-NEXT:   |   `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   |-ReturnStmt {{.*}} 
// CHECK-NEXT:   | `-CXXMemberCallExpr {{.*}} 'Task'
// CHECK-NEXT:   |   `-MemberExpr {{.*}} '<bound member function type>' .get_return_object {{.*}}
// CHECK-NEXT:   |     `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK-NEXT:   `-<<<NULL>>>