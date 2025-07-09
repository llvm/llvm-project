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
// CHECK: `-CoroutineBodyStmt {{.*}} 
// CHECK: |-CompoundStmt {{.*}} 
// CHECK: | `-ExprWithCleanups {{.*}} 'int'
// CHECK: | `-CoawaitExpr {{.*}} 'int'
// CHECK: | |-IntegerLiteral {{.*}} 'int' 1
// CHECK: | |-MaterializeTemporaryExpr {{.*}} 'awaiter' lvalue
// CHECK: | | `-CXXMemberCallExpr {{.*}} 'awaiter'
// CHECK: | | |-MemberExpr {{.*}} '<bound member function type>' .await_transform {{.*}}
// CHECK: | | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 1
// CHECK: | |-ExprWithCleanups {{.*}} 'bool'
// CHECK: | | `-CXXMemberCallExpr {{.*}} 'bool'
// CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .await_ready {{.*}}
// CHECK: | | `-OpaqueValueExpr {{.*}} 'awaiter' lvalue
// CHECK: | | `-MaterializeTemporaryExpr {{.*}} 'awaiter' lvalue
// CHECK: | | `-CXXMemberCallExpr {{.*}} 'awaiter'
// CHECK: | | |-MemberExpr {{.*}} '<bound member function type>' .await_transform {{.*}}
// CHECK: | | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 1
// CHECK: | |-ExprWithCleanups {{.*}} 'void'
// CHECK: | | `-CXXMemberCallExpr {{.*}} 'void'
// CHECK: | | |-MemberExpr {{.*}} '<bound member function type>' .await_suspend {{.*}}
// CHECK: | | | `-OpaqueValueExpr {{.*}} 'awaiter' lvalue
// CHECK: | | | `-MaterializeTemporaryExpr {{.*}} 'awaiter' lvalue
// CHECK: | | | `-CXXMemberCallExpr {{.*}} 'awaiter'
// CHECK: | | | |-MemberExpr {{.*}} '<bound member function type>' .await_transform {{.*}}
// CHECK: | | | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK: | | | `-IntegerLiteral {{.*}} 'int' 1
// CHECK: | | `-CXXConstructExpr {{.*}} 'std::coroutine_handle<>':'std::coroutine_handle<void>' 'void (coroutine_handle<void> &&) noexcept'
// CHECK: | | `-ImplicitCastExpr {{.*}} 'coroutine_handle<void>':'std::coroutine_handle<void>' xvalue <DerivedToBase (coroutine_handle)>
// CHECK: | | `-MaterializeTemporaryExpr {{.*}} 'coroutine_handle<promise_type>':'std::coroutine_handle<Task::promise_type>' xvalue
// CHECK: | | `-CallExpr {{.*}} 'coroutine_handle<promise_type>':'std::coroutine_handle<Task::promise_type>'
// CHECK: | | |-ImplicitCastExpr {{.*}} 'coroutine_handle<promise_type> (*)(void *) noexcept' <FunctionToPointerDecay>
// CHECK: | | | `-DeclRefExpr {{.*}} 'coroutine_handle<promise_type> (void *) noexcept' lvalue CXXMethod {{.*}} 'from_address' 'coroutine_handle<promise_type> (void *) noexcept'
// CHECK: | | `-CallExpr {{.*}} 'void *'
// CHECK: | | `-ImplicitCastExpr {{.*}} 'void *(*)() noexcept' <FunctionToPointerDecay>
// CHECK: | | `-DeclRefExpr {{.*}} 'void *() noexcept' lvalue Function {{.*}} '__builtin_coro_frame' 'void *() noexcept'
// CHECK: | `-CXXMemberCallExpr {{.*}} 'int'
// CHECK: | `-MemberExpr {{.*}} '<bound member function type>' .await_resume {{.*}}
// CHECK: | `-OpaqueValueExpr {{.*}} 'awaiter' lvalue
// CHECK: | `-MaterializeTemporaryExpr {{.*}} 'awaiter' lvalue
// CHECK: | `-CXXMemberCallExpr {{.*}} 'awaiter'
// CHECK: | |-MemberExpr {{.*}} '<bound member function type>' .await_transform {{.*}}
// CHECK: | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK: | `-IntegerLiteral {{.*}} 'int' 1
// CHECK: |-DeclStmt {{.*}} 
// CHECK: | `-VarDecl {{.*}} implicit used __promise 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' callinit
// CHECK: | |-CXXConstructExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' 'void () noexcept'
// CHECK: | `-typeDetails: TypedefType {{.*}} 'std::coroutine_traits<Task>::promise_type' sugar
// CHECK: | |-TypeAlias {{.*}} 'promise_type'
// CHECK: | `-typeDetails: ElaboratedType {{.*}} 'typename Task::promise_type' sugar
// CHECK: | `-typeDetails: RecordType {{.*}} 'Task::promise_type'
// CHECK: | `-CXXRecord {{.*}} 'promise_type'
// CHECK: |-ExprWithCleanups {{.*}} 'void'
// CHECK: | `-CoawaitExpr {{.*}} 'void' implicit
// CHECK: | |-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
// CHECK: | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK: | |-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK: | | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
// CHECK: | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK: | |-ExprWithCleanups {{.*}} 'bool'
// CHECK: | | `-CXXMemberCallExpr {{.*}} 'bool'
// CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .await_ready {{.*}}
// CHECK: | | `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK: | | `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK: | | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
// CHECK: | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK: | |-ExprWithCleanups {{.*}} 'void'
// CHECK: | | `-CXXMemberCallExpr {{.*}} 'void'
// CHECK: | | |-MemberExpr {{.*}} '<bound member function type>' .await_suspend {{.*}}
// CHECK: | | | `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK: | | | `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK: | | | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK: | | | `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
// CHECK: | | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK: | | `-CXXConstructExpr {{.*}} 'coroutine_handle<>':'std::coroutine_handle<void>' 'void (coroutine_handle<void> &&) noexcept'
// CHECK: | | `-ImplicitCastExpr {{.*}} 'coroutine_handle<void>':'std::coroutine_handle<void>' xvalue <DerivedToBase (coroutine_handle)>
// CHECK: | | `-MaterializeTemporaryExpr {{.*}} 'coroutine_handle<promise_type>':'std::coroutine_handle<Task::promise_type>' xvalue
// CHECK: | | `-CallExpr {{.*}} 'coroutine_handle<promise_type>':'std::coroutine_handle<Task::promise_type>'
// CHECK: | | |-ImplicitCastExpr {{.*}} 'coroutine_handle<promise_type> (*)(void *) noexcept' <FunctionToPointerDecay>
// CHECK: | | | `-DeclRefExpr {{.*}} 'coroutine_handle<promise_type> (void *) noexcept' lvalue CXXMethod {{.*}} 'from_address' 'coroutine_handle<promise_type> (void *) noexcept'
// CHECK: | | `-CallExpr {{.*}} 'void *'
// CHECK: | | `-ImplicitCastExpr {{.*}} 'void *(*)() noexcept' <FunctionToPointerDecay>
// CHECK: | | `-DeclRefExpr {{.*}} 'void *() noexcept' lvalue Function {{.*}} '__builtin_coro_frame' 'void *() noexcept'
// CHECK: | `-CXXMemberCallExpr {{.*}} 'void'
// CHECK: | `-MemberExpr {{.*}} '<bound member function type>' .await_resume {{.*}}
// CHECK: | `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK: | `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK: | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK: | `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
// CHECK: | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK: |-ExprWithCleanups {{.*}} 'void'
// CHECK: | `-CoawaitExpr {{.*}} 'void' implicit
// CHECK: | |-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
// CHECK: | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK: | |-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK: | | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
// CHECK: | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK: | |-ExprWithCleanups {{.*}} 'bool'
// CHECK: | | `-CXXMemberCallExpr {{.*}} 'bool'
// CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .await_ready {{.*}}
// CHECK: | | `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK: | | `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK: | | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
// CHECK: | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK: | |-ExprWithCleanups {{.*}} 'void'
// CHECK: | | `-CXXMemberCallExpr {{.*}} 'void'
// CHECK: | | |-MemberExpr {{.*}} '<bound member function type>' .await_suspend {{.*}}
// CHECK: | | | `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK: | | | `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK: | | | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK: | | | `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
// CHECK: | | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK: | | `-CXXConstructExpr {{.*}} 'coroutine_handle<>':'std::coroutine_handle<void>' 'void (coroutine_handle<void> &&) noexcept'
// CHECK: | | `-ImplicitCastExpr {{.*}} 'coroutine_handle<void>':'std::coroutine_handle<void>' xvalue <DerivedToBase (coroutine_handle)>
// CHECK: | | `-MaterializeTemporaryExpr {{.*}} 'coroutine_handle<promise_type>':'std::coroutine_handle<Task::promise_type>' xvalue
// CHECK: | | `-CallExpr {{.*}} 'coroutine_handle<promise_type>':'std::coroutine_handle<Task::promise_type>'
// CHECK: | | |-ImplicitCastExpr {{.*}} 'coroutine_handle<promise_type> (*)(void *) noexcept' <FunctionToPointerDecay>
// CHECK: | | | `-DeclRefExpr {{.*}} 'coroutine_handle<promise_type> (void *) noexcept' lvalue CXXMethod {{.*}} 'from_address' 'coroutine_handle<promise_type> (void *) noexcept'
// CHECK: | | `-CallExpr {{.*}} 'void *'
// CHECK: | | `-ImplicitCastExpr {{.*}} 'void *(*)() noexcept' <FunctionToPointerDecay>
// CHECK: | | `-DeclRefExpr {{.*}} 'void *() noexcept' lvalue Function {{.*}} '__builtin_coro_frame' 'void *() noexcept'
// CHECK: | `-CXXMemberCallExpr {{.*}} 'void'
// CHECK: | `-MemberExpr {{.*}} '<bound member function type>' .await_resume {{.*}}
// CHECK: | `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
// CHECK: | `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
// CHECK: | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK: | `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
// CHECK: | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK: |-<<<NULL>>>
// CHECK: |-CoreturnStmt {{.*}} implicit
// CHECK: | |-<<<NULL>>>
// CHECK: | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
// CHECK: | `-MemberExpr {{.*}} '<bound member function type>' .return_void {{.*}}
// CHECK: | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK: |-CallExpr {{.*}} 'void *'
// CHECK: | |-ImplicitCastExpr {{.*}} 'void *(*)(unsigned long)' <FunctionToPointerDecay>
// CHECK: | | `-DeclRefExpr {{.*}} 'void *(unsigned long)' lvalue Function {{.*}} 'operator new' 'void *(unsigned long)'
// CHECK: | `-CallExpr {{.*}} 'unsigned long'
// CHECK: | `-ImplicitCastExpr {{.*}} 'unsigned long (*)() noexcept' <FunctionToPointerDecay>
// CHECK: | `-DeclRefExpr {{.*}} 'unsigned long () noexcept' lvalue Function {{.*}} '__builtin_coro_size' 'unsigned long () noexcept'
// CHECK: |-CallExpr {{.*}} 'void'
// CHECK: | |-ImplicitCastExpr {{.*}} 'void (*)(void *, unsigned long) noexcept' <FunctionToPointerDecay>
// CHECK: | | `-DeclRefExpr {{.*}} 'void (void *, unsigned long) noexcept' lvalue Function {{.*}} 'operator delete' 'void (void *, unsigned long) noexcept'
// CHECK: | |-CallExpr {{.*}} 'void *'
// CHECK: | | |-ImplicitCastExpr {{.*}} 'void *(*)(void *) noexcept' <FunctionToPointerDecay>
// CHECK: | | | `-DeclRefExpr {{.*}} 'void *(void *) noexcept' lvalue Function {{.*}} '__builtin_coro_free' 'void *(void *) noexcept'
// CHECK: | | `-CallExpr {{.*}} 'void *'
// CHECK: | | `-ImplicitCastExpr {{.*}} 'void *(*)() noexcept' <FunctionToPointerDecay>
// CHECK: | | `-DeclRefExpr {{.*}} 'void *() noexcept' lvalue Function {{.*}} '__builtin_coro_frame' 'void *() noexcept'
// CHECK: | `-CallExpr {{.*}} 'unsigned long'
// CHECK: | `-ImplicitCastExpr {{.*}} 'unsigned long (*)() noexcept' <FunctionToPointerDecay>
// CHECK: | `-DeclRefExpr {{.*}} 'unsigned long () noexcept' lvalue Function {{.*}} '__builtin_coro_size' 'unsigned long () noexcept'
// CHECK: |-<<<NULL>>>
// CHECK: |-CXXMemberCallExpr {{.*}} 'Task'
// CHECK: | `-MemberExpr {{.*}} '<bound member function type>' .get_return_object {{.*}}
// CHECK: | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK: |-ReturnStmt {{.*}} 
// CHECK: | `-CXXMemberCallExpr {{.*}} 'Task'
// CHECK: | `-MemberExpr {{.*}} '<bound member function type>' .get_return_object {{.*}}
// CHECK: | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
// CHECK: `-<<<NULL>>>

Task test2() {
 co_return;
}

//CHECK: FunctionDecl {{.*}} test2 'Task ()'
//CHECK: `-CoroutineBodyStmt {{.*}} 
//CHECK: |-CompoundStmt {{.*}} 
//CHECK: | `-CoreturnStmt {{.*}} 
//CHECK: | |-<<<NULL>>>
//CHECK: | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
//CHECK: | `-MemberExpr {{.*}} '<bound member function type>' .return_void {{.*}}
//CHECK: | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
//CHECK: |-DeclStmt {{.*}} 
//CHECK: | `-VarDecl {{.*}} implicit used __promise 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' callinit
//CHECK: | |-CXXConstructExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' 'void () noexcept'
//CHECK: | `-typeDetails: TypedefType {{.*}} 'std::coroutine_traits<Task>::promise_type' sugar
//CHECK: | |-TypeAlias {{.*}} 'promise_type'
//CHECK: | `-typeDetails: ElaboratedType {{.*}} 'typename Task::promise_type' sugar
//CHECK: | `-typeDetails: RecordType {{.*}} 'Task::promise_type'
//CHECK: | `-CXXRecord {{.*}} 'promise_type'
//CHECK: |-ExprWithCleanups {{.*}} 'void'
//CHECK: | `-CoawaitExpr {{.*}} 'void' implicit
//CHECK: | |-CXXMemberCallExpr {{.*}} 'std::suspend_always'
//CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
//CHECK: | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
//CHECK: | |-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
//CHECK: | | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
//CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
//CHECK: | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
//CHECK: | |-ExprWithCleanups {{.*}} 'bool'
//CHECK: | | `-CXXMemberCallExpr {{.*}} 'bool'
//CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .await_ready {{.*}}
//CHECK: | | `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
//CHECK: | | `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
//CHECK: | | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
//CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
//CHECK: | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
//CHECK: | |-ExprWithCleanups {{.*}} 'void'
//CHECK: | | `-CXXMemberCallExpr {{.*}} 'void'
//CHECK: | | |-MemberExpr {{.*}} '<bound member function type>' .await_suspend {{.*}}
//CHECK: | | | `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
//CHECK: | | | `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
//CHECK: | | | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
//CHECK: | | | `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
//CHECK: | | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
//CHECK: | | `-CXXConstructExpr {{.*}} 'coroutine_handle<>':'std::coroutine_handle<void>' 'void (coroutine_handle<void> &&) noexcept'
//CHECK: | | `-ImplicitCastExpr {{.*}} 'coroutine_handle<void>':'std::coroutine_handle<void>' xvalue <DerivedToBase (coroutine_handle)>
//CHECK: | | `-MaterializeTemporaryExpr {{.*}} 'coroutine_handle<promise_type>':'std::coroutine_handle<Task::promise_type>' xvalue
//CHECK: | | `-CallExpr {{.*}} 'coroutine_handle<promise_type>':'std::coroutine_handle<Task::promise_type>'
//CHECK: | | |-ImplicitCastExpr {{.*}} 'coroutine_handle<promise_type> (*)(void *) noexcept' <FunctionToPointerDecay>
//CHECK: | | | `-DeclRefExpr {{.*}} 'coroutine_handle<promise_type> (void *) noexcept' lvalue CXXMethod {{.*}} 'from_address' 'coroutine_handle<promise_type> (void *) noexcept'
//CHECK: | | `-CallExpr {{.*}} 'void *'
//CHECK: | | `-ImplicitCastExpr {{.*}} 'void *(*)() noexcept' <FunctionToPointerDecay>
//CHECK: | | `-DeclRefExpr {{.*}} 'void *() noexcept' lvalue Function {{.*}} '__builtin_coro_frame' 'void *() noexcept'
//CHECK: | `-CXXMemberCallExpr {{.*}} 'void'
//CHECK: | `-MemberExpr {{.*}} '<bound member function type>' .await_resume {{.*}}
//CHECK: | `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
//CHECK: | `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
//CHECK: | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
//CHECK: | `-MemberExpr {{.*}} '<bound member function type>' .initial_suspend {{.*}}
//CHECK: | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
//CHECK: |-ExprWithCleanups {{.*}} 'void'
//CHECK: | `-CoawaitExpr {{.*}} 'void' implicit
//CHECK: | |-CXXMemberCallExpr {{.*}} 'std::suspend_always'
//CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
//CHECK: | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
//CHECK: | |-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
//CHECK: | | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
//CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
//CHECK: | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
//CHECK: | |-ExprWithCleanups {{.*}} 'bool'
//CHECK: | | `-CXXMemberCallExpr {{.*}} 'bool'
//CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .await_ready {{.*}}
//CHECK: | | `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
//CHECK: | | `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
//CHECK: | | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
//CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
//CHECK: | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
//CHECK: | |-ExprWithCleanups {{.*}} 'void'
//CHECK: | | `-CXXMemberCallExpr {{.*}} 'void'
//CHECK: | | |-MemberExpr {{.*}} '<bound member function type>' .await_suspend {{.*}}
//CHECK: | | | `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
//CHECK: | | | `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
//CHECK: | | | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
//CHECK: | | | `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
//CHECK: | | | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
//CHECK: | | `-CXXConstructExpr {{.*}} 'coroutine_handle<>':'std::coroutine_handle<void>' 'void (coroutine_handle<void> &&) noexcept'
//CHECK: | | `-ImplicitCastExpr {{.*}} 'coroutine_handle<void>':'std::coroutine_handle<void>' xvalue <DerivedToBase (coroutine_handle)>
//CHECK: | | `-MaterializeTemporaryExpr {{.*}} 'coroutine_handle<promise_type>':'std::coroutine_handle<Task::promise_type>' xvalue
//CHECK: | | `-CallExpr {{.*}} 'coroutine_handle<promise_type>':'std::coroutine_handle<Task::promise_type>'
//CHECK: | | |-ImplicitCastExpr {{.*}} 'coroutine_handle<promise_type> (*)(void *) noexcept' <FunctionToPointerDecay>
//CHECK: | | | `-DeclRefExpr {{.*}} 'coroutine_handle<promise_type> (void *) noexcept' lvalue CXXMethod {{.*}} 'from_address' 'coroutine_handle<promise_type> (void *) noexcept'
//CHECK: | | `-CallExpr {{.*}} 'void *'
//CHECK: | | `-ImplicitCastExpr {{.*}} 'void *(*)() noexcept' <FunctionToPointerDecay>
//CHECK: | | `-DeclRefExpr {{.*}} 'void *() noexcept' lvalue Function {{.*}} '__builtin_coro_frame' 'void *() noexcept'
//CHECK: | `-CXXMemberCallExpr {{.*}} 'void'
//CHECK: | `-MemberExpr {{.*}} '<bound member function type>' .await_resume {{.*}}
//CHECK: | `-OpaqueValueExpr {{.*}} 'std::suspend_always' lvalue
//CHECK: | `-MaterializeTemporaryExpr {{.*}} 'std::suspend_always' lvalue
//CHECK: | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
//CHECK: | `-MemberExpr {{.*}} '<bound member function type>' .final_suspend {{.*}}
//CHECK: | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
//CHECK: |-<<<NULL>>>
//CHECK: |-CoreturnStmt {{.*}} implicit
//CHECK: | |-<<<NULL>>>
//CHECK: | `-CXXMemberCallExpr {{.*}} 'std::suspend_always'
//CHECK: | `-MemberExpr {{.*}} '<bound member function type>' .return_void {{.*}}
//CHECK: | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
//CHECK: |-CallExpr {{.*}} 'void *'
//CHECK: | |-ImplicitCastExpr {{.*}} 'void *(*)(unsigned long)' <FunctionToPointerDecay>
//CHECK: | | `-DeclRefExpr {{.*}} 'void *(unsigned long)' lvalue Function {{.*}} 'operator new' 'void *(unsigned long)'
//CHECK: | `-CallExpr {{.*}} 'unsigned long'
//CHECK: | `-ImplicitCastExpr {{.*}} 'unsigned long (*)() noexcept' <FunctionToPointerDecay>
//CHECK: | `-DeclRefExpr {{.*}} 'unsigned long () noexcept' lvalue Function {{.*}} '__builtin_coro_size' 'unsigned long () noexcept'
//CHECK: |-CallExpr {{.*}} 'void'
//CHECK: | |-ImplicitCastExpr {{.*}} 'void (*)(void *, unsigned long) noexcept' <FunctionToPointerDecay>
//CHECK: | | `-DeclRefExpr {{.*}} 'void (void *, unsigned long) noexcept' lvalue Function {{.*}} 'operator delete' 'void (void *, unsigned long) noexcept'
//CHECK: | |-CallExpr {{.*}} 'void *'
//CHECK: | | |-ImplicitCastExpr {{.*}} 'void *(*)(void *) noexcept' <FunctionToPointerDecay>
//CHECK: | | | `-DeclRefExpr {{.*}} 'void *(void *) noexcept' lvalue Function {{.*}} '__builtin_coro_free' 'void *(void *) noexcept'
//CHECK: | | `-CallExpr {{.*}} 'void *'
//CHECK: | | `-ImplicitCastExpr {{.*}} 'void *(*)() noexcept' <FunctionToPointerDecay>
//CHECK: | | `-DeclRefExpr {{.*}} 'void *() noexcept' lvalue Function {{.*}} '__builtin_coro_frame' 'void *() noexcept'
//CHECK: | `-CallExpr {{.*}} 'unsigned long'
//CHECK: | `-ImplicitCastExpr {{.*}} 'unsigned long (*)() noexcept' <FunctionToPointerDecay>
//CHECK: | `-DeclRefExpr {{.*}} 'unsigned long () noexcept' lvalue Function {{.*}} '__builtin_coro_size' 'unsigned long () noexcept'
//CHECK: |-<<<NULL>>>
//CHECK: |-CXXMemberCallExpr {{.*}} 'Task'
//CHECK: | `-MemberExpr {{.*}} '<bound member function type>' .get_return_object {{.*}}
//CHECK: | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
//CHECK: |-ReturnStmt {{.*}} 
//CHECK: | `-CXXMemberCallExpr {{.*}} 'Task'
//CHECK: | `-MemberExpr {{.*}} '<bound member function type>' .get_return_object {{.*}}
//CHECK: | `-DeclRefExpr {{.*}} 'std::coroutine_traits<Task>::promise_type':'Task::promise_type' lvalue Var {{.*}} '__promise' 'std::coroutine_traits<Task>::promise_type':'Task::promise_type'
//CHECK: `-<<<NULL>>>