// RUN: %clang_cc1 -std=c++2d -fsyntax-only  -embed-dir=%S/Inputs %s

namespace std {
  enum byte : unsigned char {};
  typedef decltype(sizeof(0)) size_t;
}

consteval void f () {

  unsigned int locus = 0;
  int status = 0;
  std::size_t size = 0;
  const char* ptrc = 0;
  std::size_t offset = 0;
  std::size_t limit = 0;
  struct illegal_t { unsigned char c; } illegal = {};

  (void)__builtin_std_embed(locus, status, size, ptrc, 0, ptrc, offset, limit, illegal);
  // error@-1 {{ too many arguments to function call, expected at most 8, have 9 }}
  (void)__builtin_std_embed(locus, status, size);
  // error@-1 {{ too few arguments to function call, expected 7, have 3 }}
  (void)__builtin_std_embed(locus, status, size, ptrc, 0, ptrc);
  // error@-1 {{ too few arguments to function call, expected 7, have 6 }}
  (void)__builtin_std_embed(illegal, status, size, ptrc, 0, ptrc, offset, limit);
  // error@-1 {{ no viable conversion from 'struct illegal_t' to 'unsigned int' }}
  (void)__builtin_std_embed(locus, illegal, size, ptrc, 0, ptrc, offset, limit);
  // error@-1 {{ non-const lvalue reference to type 'int' cannot bind to a value of unrelated type 'struct illegal_t' }}
  (void)__builtin_std_embed(locus, status, illegal, ptrc, 0, ptrc, offset, limit);
  // error@-1 {{ non-const lvalue reference to type '__size_t' (aka 'unsigned long') cannot bind to a value of unrelated type 'struct illegal_t' }}
  (void)__builtin_std_embed(locus, status, size, illegal, 0, ptrc, offset, limit);
  // error@-1 {{ invalid argument to '__builtin_std_embed': 'struct illegal_t' should be a pointer to const 'char', 'unsigned char', or 'std::byte' }}
  (void)__builtin_std_embed(locus, status, size, ptrc, illegal, ptrc, offset, limit);
  // error@-1 {{ invalid argument to '__builtin_std_embed': 'struct illegal_t' should be an integral type with a non-negative value }}
  (void)__builtin_std_embed(locus, status, size, ptrc, 0, illegal, offset, limit);
  // error@-1 {{ invalid argument to '__builtin_std_embed': 'struct illegal_t' should be a pointer to (possibly qualified) 'char', 'wchar_t', or 'char8_t' }}
  (void)__builtin_std_embed(locus, status, size, ptrc, 0, ptrc, illegal, limit);
  // error@-1 {{ value of type 'struct illegal_t' is not implicitly convertible to a non-negative integer of integral type }}
  (void)__builtin_std_embed(locus, status, size, ptrc, 0, ptrc, offset, illegal);
  // error@-1 {{ value of type 'struct illegal_t' is not implicitly convertible to a non-negative integer of integral type }}
}
