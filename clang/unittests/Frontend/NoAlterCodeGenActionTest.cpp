//===- unittests/Frontend/NoAlterCodeGenActionTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for CodeGenAction may not alter the AST.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/LangStandard.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;
using namespace clang::frontend;
using namespace clang::tooling;

namespace {

class ASTChecker : public RecursiveASTVisitor<ASTChecker> {
public:
  ASTContext &Ctx;
  ASTChecker(ASTContext &Ctx) : Ctx(Ctx) {}
  bool VisitReturnStmt(ReturnStmt *RS) {
    EXPECT_TRUE(RS->getRetValue());
    return true;
  }

  bool VisitCoroutineBodyStmt(CoroutineBodyStmt *CS) {
    return VisitReturnStmt(cast<ReturnStmt>(CS->getReturnStmt()));
  }
};

class ASTCheckerConsumer : public ASTConsumer {
public:
  void HandleTranslationUnit(ASTContext &Ctx) override {
    ASTChecker Checker(Ctx);
    Checker.TraverseAST(Ctx);
  }
};

class TestCodeGenAction : public EmitLLVMOnlyAction {
public:
  using Base = EmitLLVMOnlyAction;
  using Base::Base;

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    std::vector<std::unique_ptr<ASTConsumer>> Consumers;
    Consumers.push_back(std::make_unique<ASTCheckerConsumer>());
    Consumers.push_back(Base::CreateASTConsumer(CI, InFile));
    return std::make_unique<MultiplexConsumer>(std::move(Consumers));
  }
};

const char *test_contents = R"cpp(

namespace std {

template <typename R, typename...> struct coroutine_traits {
  using promise_type = typename R::promise_type;
};

template <typename Promise = void> struct coroutine_handle;

template <> struct coroutine_handle<void> {
  static coroutine_handle from_address(void *addr) noexcept;
  void operator()() { resume(); }
  void *address() const noexcept;
  void resume() const { __builtin_coro_resume(ptr); }
  void destroy() const { __builtin_coro_destroy(ptr); }
  bool done() const;
  coroutine_handle &operator=(decltype(nullptr));
  coroutine_handle(decltype(nullptr)) : ptr(nullptr) {}
  coroutine_handle() : ptr(nullptr) {}
//  void reset() { ptr = nullptr; } // add to P0057?
  explicit operator bool() const;

protected:
  void *ptr;
};

template <typename Promise> struct coroutine_handle : coroutine_handle<> {
  using coroutine_handle<>::operator=;

  static coroutine_handle from_address(void *addr) noexcept;

  Promise &promise() const;
  static coroutine_handle from_promise(Promise &promise);
};

template <typename _PromiseT>
bool operator==(coroutine_handle<_PromiseT> const &_Left,
                coroutine_handle<_PromiseT> const &_Right) noexcept {
  return _Left.address() == _Right.address();
}

template <typename _PromiseT>
bool operator!=(coroutine_handle<_PromiseT> const &_Left,
                coroutine_handle<_PromiseT> const &_Right) noexcept {
  return !(_Left == _Right);
}

struct noop_coroutine_promise {};

template <>
struct coroutine_handle<noop_coroutine_promise> {
  operator coroutine_handle<>() const noexcept;

  constexpr explicit operator bool() const noexcept { return true; }
  constexpr bool done() const noexcept { return false; }

  constexpr void operator()() const noexcept {}
  constexpr void resume() const noexcept {}
  constexpr void destroy() const noexcept {}

  noop_coroutine_promise &promise() const noexcept {
    return *static_cast<noop_coroutine_promise *>(
        __builtin_coro_promise(this->__handle_, alignof(noop_coroutine_promise), false));
  }

  constexpr void *address() const noexcept { return __handle_; }

private:
  friend coroutine_handle<noop_coroutine_promise> noop_coroutine() noexcept;

  coroutine_handle() noexcept {
    this->__handle_ = __builtin_coro_noop();
  }

  void *__handle_ = nullptr;
};

using noop_coroutine_handle = coroutine_handle<noop_coroutine_promise>;

inline noop_coroutine_handle noop_coroutine() noexcept { return noop_coroutine_handle(); }

struct suspend_always {
  bool await_ready() noexcept { return false; }
  void await_suspend(coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};
struct suspend_never {
  bool await_ready() noexcept { return true; }
  void await_suspend(coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

} // namespace std

using namespace std;

class invoker {
public:
  class invoker_promise {
  public:
    invoker get_return_object() { return invoker{}; }
    auto initial_suspend() { return suspend_always{}; }
    auto final_suspend() noexcept { return suspend_always{}; }
    void return_void() {}
    void unhandled_exception() {}
  };
  using promise_type = invoker_promise;
  invoker() {}
  invoker(const invoker &) = delete;
  invoker &operator=(const invoker &) = delete;
  invoker(invoker &&) = delete;
  invoker &operator=(invoker &&) = delete;
};

invoker g() {
  co_return;
}

)cpp";

TEST(CodeGenTest, TestNonAlterTest) {
  EXPECT_TRUE(runToolOnCodeWithArgs(std::make_unique<TestCodeGenAction>(),
                                    test_contents,
                                    {
                                        "-std=c++20",
                                    }));
}
} // namespace
