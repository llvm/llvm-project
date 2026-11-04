// RUN: %clang_cc1 -fptrauth-intrinsics -fsyntax-only -ferror-limit 1 -verify -std=c++26 %s
// RUN: %clang_cc1 -fptrauth-intrinsics -fsyntax-only -ferror-limit 1 -verify -std=c++03 %s
// RUN: %clang_cc1                      -fsyntax-only -ferror-limit 1 -verify -std=c++03 %s

/// Force two errors so we hit the error limit leading to skip of template instantiation
# "" // expected-error {{invalid preprocessing directive}}
# ""
// expected-error@* {{too many errors emitted}}

template <typename>
struct a {};
struct test_polymorphic {
  virtual ~test_polymorphic();
  a<int> field;
};
static_assert(__is_trivially_relocatable(test_polymorphic));

struct test_struct {
  test_struct(int) {}
  void test_instantiate() {
    test_struct d(0);
  }
  void test_type_trait_query() {
    __is_trivially_relocatable(test_struct);
  }
  a<int> e;
};

struct test_struct2 {
  test_struct member;
  void test() {
    test_struct2 t{.member = {0}};
  }
};

struct test_subclass : test_struct {
   test_subclass() : test_struct(0) {
   }

   void test_subclass_instantiation() {
    test_subclass subclass{};
   }
  void test_subclass_type_trait_query() {
    __is_trivially_relocatable(test_subclass);
  }
};
