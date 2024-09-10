// RUN: %clang_cc1 -std=c++20 -ast-dump %s | FileCheck %s
#include "Inputs/std-coroutine.h"

// Github issue: https://github.com/llvm/llvm-project/issues/78290
namespace GH78290 {
class Gen {
   public:
    class promise_type {
       public:
        template<typename... Args>
        explicit promise_type(Args...) {}
        // CHECK:       CXXConstructorDecl {{.*}} used promise_type 'void ()' {{.*}}
        // CHECK-NEXT:     TemplateArgument pack
        // CHECK-NEXT:     CompoundStmt {{.*}}
        Gen get_return_object() { return {}; }

        void unhandled_exception() {}
        void return_void() {}
        std::suspend_always await_transform(Gen gen) { return {}; }

        std::suspend_always initial_suspend() { return {}; }
        // CHECK: CXXMethodDecl {{.*}} used initial_suspend {{.*}}
        std::suspend_always final_suspend() noexcept { return {}; }
        // CHECK: CXXMethodDecl {{.*}} used final_suspend {{.*}}
    };
};

Gen CoroutineBody() {
    if constexpr (0) {
        co_await Gen{};
    }
    co_await Gen{};
}
} // namespace GH78290
