// RUN: %check_clang_tidy %s misc-use-internal-linkage %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     misc-use-internal-linkage.AnalyzeVariables: false, \
// RUN:     misc-use-internal-linkage.AnalyzeTypes: false \
// RUN:   }}" -- -I%S/Inputs/use-internal-linkage

// RUN: %check_clang_tidy %s misc-use-internal-linkage %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     misc-use-internal-linkage.FixMode: 'UseStatic', \
// RUN:     misc-use-internal-linkage.AnalyzeVariables: false, \
// RUN:     misc-use-internal-linkage.AnalyzeTypes: false \
// RUN:   }}" -- -I%S/Inputs/use-internal-linkage

#include "func.h"

void func() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'func' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static void func() {}

template<class T>
void func_template() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'func_template' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static void func_template() {}

void func_cpp_inc() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'func_cpp_inc' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static void func_cpp_inc() {}

int* func_cpp_inc_return_ptr() { return nullptr; }
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'func_cpp_inc_return_ptr' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static int* func_cpp_inc_return_ptr() { return nullptr; }

const int* func_cpp_inc_return_const_ptr() { return nullptr; }
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: function 'func_cpp_inc_return_const_ptr' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static const int* func_cpp_inc_return_const_ptr() { return nullptr; }

int const* func_cpp_inc_return_ptr_const() { return nullptr; }
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: function 'func_cpp_inc_return_ptr_const' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static int const* func_cpp_inc_return_ptr_const() { return nullptr; }

int * const func_cpp_inc_return_const() { return nullptr; }
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: function 'func_cpp_inc_return_const' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static int * const func_cpp_inc_return_const() { return nullptr; }

volatile const int* func_cpp_inc_return_volatile_const_ptr() { return nullptr; }
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: function 'func_cpp_inc_return_volatile_const_ptr' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static volatile const int* func_cpp_inc_return_volatile_const_ptr() { return nullptr; }

[[nodiscard]] void func_nodiscard() {}
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: function 'func_nodiscard' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: {{\[\[nodiscard\]\]}} static void func_nodiscard() {}

#define NDS [[nodiscard]]
#define NNDS

NDS void func_nds() {}
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: function 'func_nds' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: NDS static void func_nds() {}

NNDS void func_nnds() {}
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: function 'func_nnds' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: NNDS static void func_nnds() {}

#include "func_cpp.inc"

void func_h_inc() {}

struct S {
  void method();
};
void S::method() {}

void func_header() {}
extern void func_extern() {}
static void func_static() {}
namespace {
void func_anonymous_ns() {}
} // namespace

int main(int argc, const char*argv[]) {}

extern "C" {
void func_extern_c_1() {}
}

extern "C" void func_extern_c_2() {}

namespace gh117488 {
void func_with_body();
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'func_with_body' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static void func_with_body();
void func_with_body() {}

void func_without_body();
void func_without_body();
}

// gh117489 start
namespace std {
using size_t = decltype(sizeof(int));
}
void * operator new(std::size_t) { return nullptr; }
void * operator new[](std::size_t) { return nullptr; }
void operator delete(void*) noexcept {}
void operator delete[](void*) noexcept {}
// gh117489 end

int ignored_global = 0; // AnalyzeVariables is false.
struct IgnoredStruct {}; // AnalyzeTypes is false.
