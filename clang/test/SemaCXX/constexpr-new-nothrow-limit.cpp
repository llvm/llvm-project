// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fconstexpr-steps=1024 -verify=both %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fconstexpr-steps=1024 -verify=both -fexperimental-new-constant-interpreter %s

namespace std {
using size_t = decltype(sizeof(0));
struct nothrow_t {};
inline constexpr nothrow_t nothrow{};
} // namespace std

void *operator new[](std::size_t, const std::nothrow_t &) noexcept;
void operator delete[](void *) noexcept;

// Past -fconstexpr-steps the evaluator has decided nothing about this
// allocation, so it is not a constant. It used to come out as a null pointer:
// the evaluator's own resource limit was reported like an erroneous array
// bound, and [expr.new] turns an erroneous non-throwing new-expression into a
// null pointer, which counted as a successful result with no diagnostic. Both
// interpreters now agree.
constexpr auto overLimit = new (std::nothrow) unsigned char[1025];
// both-error@-1 {{constexpr variable 'overLimit' must be initialized by a constant expression}}
// both-note@-2 {{cannot allocate array; evaluated array bound 1025 exceeds the limit (1024)}}
// both-note@-3 {{use -fconstexpr-steps= to increase this limit (=0 will remove the limit)}}

// A throwing allocation past the limit always failed and explained itself.
constexpr auto throwingOverLimit = new unsigned char[1025];
// both-error@-1 {{constexpr variable 'throwingOverLimit' must be initialized by a constant expression}}
// both-note@-2 {{cannot allocate array; evaluated array bound 1025 exceeds the limit (1024)}}
// both-note@-3 {{use -fconstexpr-steps= to increase this limit (=0 will remove the limit)}}

// A bound that no address space can hold is an erroneous new-expression, so a
// non-throwing allocation of it is a null pointer, without a diagnostic.
constexpr auto tooBig = new (std::nothrow) unsigned char[4294967296ULL];
static_assert(tooBig == nullptr);
