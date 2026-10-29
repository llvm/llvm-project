// RUN: %check_clang_tidy -std=c++17-or-later %s bugprone-std-namespace-modification %t \
// RUN:   -- -system-headers

// Compiler-generated implicit declarations in namespace std (e.g.
// std::align_val_t) should not trigger a warning or crash clang-tidy.
// A new expression forces Clang to implicitly declare std::align_val_t
// inside namespace std when alignment is enabled. This test verifies
// the checker handles this situation correctly in case of -system-headers
// switch present. Situation without -system-headers is handled in
// std-namespace-modification.cpp test file.

namespace std {}

// Trigger implicit std::align_val_t declaration.
void *implicit_decl_test = new int;

namespace std {
// CHECK-MESSAGES: :[[@LINE+2]]:5: warning: modification of 'std' namespace
// CHECK-MESSAGES: :[[@LINE-2]]:11: note: 'std' namespace opened here
int x;
}
