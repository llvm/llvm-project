// Tests that flow-sensitive nullability recognizes known STL methods as
// returning nonnull, suppressing false positives from unannotated return types.
//
// Uses real system headers so AST structure matches production code.
// UNSUPPORTED: target={{.*-windows.*}}
// REQUIRES: system-darwin || system-linux
// RUN: %clangxx -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 %s -Xclang -verify

#include <vector>
#include <string>
#include <string_view>
#include <optional>

// --- std::vector::data() returns nonnull ---

void vector_data_deref() {
    std::vector<int> v = {1, 2, 3};
    int *p = v.data();
    *p = 42; // no-warning: data() is in the STL nonnull allowlist
}

// --- std::string::c_str() and data() return nonnull ---

void string_c_str_deref() {
    std::string s = "hello";
    const char *p = s.c_str();
    char c = *p; // no-warning: c_str() is in the STL nonnull allowlist
}

void string_data_deref() {
    std::string s = "hello";
    const char *p = s.data();
    char c = *p; // no-warning: data() is in the STL nonnull allowlist
}

// --- std::string_view::data() is intentionally nullable ---

void string_view_data_warns() {
    std::string_view sv = "hello";
    const char *p = sv.data();
    char c = *p; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// --- std::optional::operator->() returns nonnull (UB if empty) ---

void optional_arrow_deref() {
    std::optional<int> opt = 42;
    int *p = opt.operator->();
    *p = 1; // no-warning: operator->() is in the STL nonnull allowlist
}

// --- Non-std methods still warn ---

struct MyContainer {
    int *get();
};

void non_std_method_warns() {
    MyContainer c;
    int *p = c.get();
    *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}
