//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: host-has-gdb-with-python
// REQUIRES: locale.en_US.UTF-8
// UNSUPPORTED: libcpp-has-no-localization
// UNSUPPORTED: c++03

// TODO: Investigate this failure, which happens only with the Bootstrapping build.
// UNSUPPORTED: clang-14

// RUN: %{cxx} %{flags} %s -o %t.exe %{compile_flags} -g %{link_flags}
// Ensure locale-independence for unicode tests.
// RUN: env LANG=en_US.UTF-8 %{gdb} -nx -batch -iex "set autoload off" -ex "source %S/../../../utils/gdb/libcxx/printers.py" -ex "python register_libcxx_printer_loader()" -ex "source %S/gdb_pretty_printer_test.py" %t.exe

#include <bitset>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "test_macros.h"

// To write a pretty-printer test:
//
// 1. Declare a variable of the type you want to test
//
// 2. Set its value to something which will test the pretty printer in an
//    interesting way.
//
// 3. Call ComparePrettyPrintToChars with that variable, and a "const char*"
//    value to compare to the printer's output.
//
//    Or
//
//    Call ComparePrettyPrintToRegex with that variable, and a "const char*"
//    *python* regular expression to match against the printer's output.
//    The set of special characters in a Python regular expression overlaps
//    with a lot of things the pretty printers print--brackets, for
//    example--so take care to escape appropriately.
//
// Alternatively, construct a string that gdb can parse as an expression,
// so that printing the value of the expression will test the pretty printer
// in an interesting way. Then, call CompareExpressionPrettyPrintToChars or
// CompareExpressionPrettyPrintToRegex to compare the printer's output.

// Avoids setting a breakpoint in every-single instantiation of
// ComparePrettyPrintTo*.  Also, make sure neither it, nor the
// variables we need present in the Compare functions are optimized
// away.
#ifdef TEST_COMPILER_GCC
#define OPT_NONE __attribute__((noinline))
#else
#define OPT_NONE __attribute__((optnone))
#endif
void StopForDebugger(void *, void *) OPT_NONE;
void StopForDebugger(void *, void *)  {}


// Prevents the compiler optimizing away the parameter in the caller function.
template <typename Type>
void MarkAsLive(Type &&) OPT_NONE;
template <typename Type>
void MarkAsLive(Type &&) {}

// In all of the Compare(Expression)PrettyPrintTo(Regex/Chars) functions below,
// the python script sets a breakpoint just before the call to StopForDebugger,
// compares the result to the expectation.
//
// The expectation is a literal string to be matched exactly in
// *PrettyPrintToChars functions, and is a python regular expression in
// *PrettyPrintToRegex functions.
//
// In ComparePrettyPrint* functions, the value is a variable of any type. In
// CompareExpressionPrettyPrint functions, the value is a string expression that
// gdb will parse and print the result.
//
// The python script will print either "PASS", or a detailed failure explanation
// along with the line that has invoke the function. The testing will continue
// in either case.

template <typename TypeToPrint> void ComparePrettyPrintToChars(
    TypeToPrint value,
    const char *expectation) {
  MarkAsLive(value);
  StopForDebugger(&value, &expectation);
}

template <typename TypeToPrint> void ComparePrettyPrintToRegex(
    TypeToPrint value,
    const char *expectation) {
  MarkAsLive(value);
  StopForDebugger(&value, &expectation);
}

void CompareExpressionPrettyPrintToChars(
    std::string value,
    const char *expectation) {
  MarkAsLive(value);
  StopForDebugger(&value, &expectation);
}

void CompareExpressionPrettyPrintToRegex(
    std::string value,
    const char *expectation) {
  MarkAsLive(value);
  StopForDebugger(&value, &expectation);
}

namespace example {
  struct example_struct {
    int a = 0;
    int arr[1000];
  };
}

// If enabled, the self test will "fail"--because we want to be sure it properly
// diagnoses tests that *should* fail. Evaluate the output by hand.
void framework_self_test() {
#ifdef FRAMEWORK_SELF_TEST
  // Use the most simple data structure we can.
  const char a = 'a';

  // Tests that should pass
  ComparePrettyPrintToChars(a, "97 'a'");
  ComparePrettyPrintToRegex(a, ".*");

  // Tests that should fail.
  ComparePrettyPrintToChars(a, "b");
  ComparePrettyPrintToRegex(a, "b");
#endif
}

// A simple pass-through allocator to check that we handle CompressedPair
// correctly.
template <typename T> class UncompressibleAllocator : public std::allocator<T> {
 public:
  char X;
};

void string_test() {
  std::string short_string("kdjflskdjf");
  // The display_hint "string" adds quotes the printed result.
  ComparePrettyPrintToChars(short_string, "\"kdjflskdjf\"");

  std::basic_string<char, std::char_traits<char>, UncompressibleAllocator<char>>
      long_string("mehmet bizim dostumuz agzi kirik testimiz");
  ComparePrettyPrintToChars(long_string,
                            "\"mehmet bizim dostumuz agzi kirik testimiz\"");
}

namespace a_namespace {
// To test name-lookup in the presence of using inside a namespace. Inside this
// namespace, unqualified string_view variables will appear in the debug info as
// "a_namespace::string_view, rather than "std::string_view".
//
// There is nothing special here about string_view; it's just the data structure
// where lookup with using inside a namespace wasn't always working.

using string_view = std::string_view;

void string_view_test() {
  std::string_view i_am_empty;
  ComparePrettyPrintToChars(i_am_empty, "\"\"");

  std::string source_string("to be or not to be");
  std::string_view to_be(source_string);
  ComparePrettyPrintToChars(to_be, "\"to be or not to be\"");

  const char char_arr[] = "what a wonderful world";
  std::string_view wonderful(&char_arr[7], 9);
  ComparePrettyPrintToChars(wonderful, "\"wonderful\"");

  const char char_arr1[] = "namespace_stringview";
  string_view namespace_stringview(&char_arr1[10], 10);
  ComparePrettyPrintToChars(namespace_stringview, "\"stringview\"");
}
}

void u16string_test() {
  std::u16string test0 = u"Hello World";
  ComparePrettyPrintToChars(test0, "u\"Hello World\"");
  std::u16string test1 = u"\U00010196\u20AC\u00A3\u0024";
  ComparePrettyPrintToChars(test1, "u\"\U00010196\u20AC\u00A3\u0024\"");
  std::u16string test2 = u"\u0024\u0025\u0026\u0027";
  ComparePrettyPrintToChars(test2, "u\"\u0024\u0025\u0026\u0027\"");
  std::u16string test3 = u"mehmet bizim dostumuz agzi kirik testimiz";
  ComparePrettyPrintToChars(test3,
                            ("u\"mehmet bizim dostumuz agzi kirik testimiz\""));
}

void u32string_test() {
  std::u32string test0 = U"Hello World";
  ComparePrettyPrintToChars(test0, "U\"Hello World\"");
  std::u32string test1 =
      U"\U0001d552\U0001d553\U0001d554\U0001d555\U0001d556\U0001d557";
  ComparePrettyPrintToChars(
      test1,
      ("U\"\U0001d552\U0001d553\U0001d554\U0001d555\U0001d556\U0001d557\""));
  std::u32string test2 = U"\U00004f60\U0000597d";
  ComparePrettyPrintToChars(test2, ("U\"\U00004f60\U0000597d\""));
  std::u32string test3 = U"mehmet bizim dostumuz agzi kirik testimiz";
  ComparePrettyPrintToChars(test3, ("U\"mehmet bizim dostumuz agzi kirik testimiz\""));
}

void tuple_test() {
  std::tuple<int, int, int> test0(2, 3, 4);
  ComparePrettyPrintToChars(
      test0,
      "std::tuple containing = {[1] = 2, [2] = 3, [3] = 4}");

  std::tuple<> test1;
  ComparePrettyPrintToChars(
      test1,
      "empty std::tuple");
}

void unique_ptr_test() {
  std::unique_ptr<std::string> matilda(new std::string("Matilda"));
  ComparePrettyPrintToRegex(
      std::move(matilda),
      R"(std::unique_ptr<std::string> containing = {__ptr_ = 0x[a-f0-9]+})");
  std::unique_ptr<int> forty_two(new int(42));
  ComparePrettyPrintToRegex(std::move(forty_two),
      R"(std::unique_ptr<int> containing = {__ptr_ = 0x[a-f0-9]+})");

  std::unique_ptr<int> this_is_null;
  ComparePrettyPrintToChars(std::move(this_is_null),
      R"(std::unique_ptr is nullptr)");
}

void bitset_test() {
  std::bitset<258> i_am_empty(0);
  ComparePrettyPrintToRegex(i_am_empty, "std::bitset<258(u|ul)?>");

  std::bitset<0> very_empty;
  ComparePrettyPrintToRegex(very_empty, "std::bitset<0(u|ul)?>");

  std::bitset<15> b_000001111111100(1020);
  ComparePrettyPrintToRegex(b_000001111111100,
      R"(std::bitset<15(u|ul)?> = {\[2\] = 1, \[3\] = 1, \[4\] = 1, \[5\] = 1, \[6\] = 1, )"
      R"(\[7\] = 1, \[8\] = 1, \[9\] = 1})");

  std::bitset<258> b_0_129_132(0);
  b_0_129_132[0] = true;
  b_0_129_132[129] = true;
  b_0_129_132[132] = true;
  ComparePrettyPrintToRegex(b_0_129_132,
      R"(std::bitset<258(u|ul)?> = {\[0\] = 1, \[129\] = 1, \[132\] = 1})");
}

void list_test() {
  std::list<int> i_am_empty{};
  ComparePrettyPrintToChars(i_am_empty, "std::list is empty");

  std::list<int> one_two_three {1, 2, 3};
  ComparePrettyPrintToChars(one_two_three,
      "std::list with 3 elements = {1, 2, 3}");

  std::list<std::string> colors {"red", "blue", "green"};
  ComparePrettyPrintToChars(colors,
      R"(std::list with 3 elements = {"red", "blue", "green"})");
}

void deque_test() {
  std::deque<int> i_am_empty{};
  ComparePrettyPrintToChars(i_am_empty, "std::deque is empty");

  std::deque<int> one_two_three {1, 2, 3};
  ComparePrettyPrintToChars(one_two_three,
      "std::deque with 3 elements = {1, 2, 3}");

  std::deque<example::example_struct> bfg;
  for (int i = 0; i < 10; ++i) {
    example::example_struct current;
    current.a = i;
    bfg.push_back(current);
  }
  for (int i = 0; i < 3; ++i) {
    bfg.pop_front();
  }
  for (int i = 0; i < 3; ++i) {
    bfg.pop_back();
  }
  ComparePrettyPrintToRegex(bfg,
      "std::deque with 4 elements = {"
      "{a = 3, arr = {[^}]+}}, "
      "{a = 4, arr = {[^}]+}}, "
      "{a = 5, arr = {[^}]+}}, "
      "{a = 6, arr = {[^}]+}}}");
}

void map_test() {
  std::map<int, int> i_am_empty{};
  ComparePrettyPrintToChars(i_am_empty, "std::map is empty");

  std::map<int, std::string> one_two_three;
  one_two_three.insert({1, "one"});
  one_two_three.insert({2, "two"});
  one_two_three.insert({3, "three"});
  ComparePrettyPrintToChars(one_two_three,
      "std::map with 3 elements = "
      R"({[1] = "one", [2] = "two", [3] = "three"})");

  std::map<int, example::example_struct> bfg;
  for (int i = 0; i < 4; ++i) {
    example::example_struct current;
    current.a = 17 * i;
    bfg.insert({i, current});
  }
  ComparePrettyPrintToRegex(bfg,
      R"(std::map with 4 elements = {)"
      R"(\[0\] = {a = 0, arr = {[^}]+}}, )"
      R"(\[1\] = {a = 17, arr = {[^}]+}}, )"
      R"(\[2\] = {a = 34, arr = {[^}]+}}, )"
      R"(\[3\] = {a = 51, arr = {[^}]+}}})");
}

void multimap_test() {
  std::multimap<int, int> i_am_empty{};
  ComparePrettyPrintToChars(i_am_empty, "std::multimap is empty");

  std::multimap<int, std::string> one_two_three;
  one_two_three.insert({1, "one"});
  one_two_three.insert({3, "three"});
  one_two_three.insert({1, "ein"});
  one_two_three.insert({2, "two"});
  one_two_three.insert({2, "zwei"});
  one_two_three.insert({1, "bir"});

  ComparePrettyPrintToChars(one_two_three,
      "std::multimap with 6 elements = "
      R"({[1] = "one", [1] = "ein", [1] = "bir", )"
      R"([2] = "two", [2] = "zwei", [3] = "three"})");
}

void queue_test() {
  std::queue<int> i_am_empty;
  ComparePrettyPrintToChars(i_am_empty,
      "std::queue wrapping = {std::deque is empty}");

  std::queue<int> one_two_three(std::deque<int>{1, 2, 3});
    ComparePrettyPrintToChars(one_two_three,
        "std::queue wrapping = {"
        "std::deque with 3 elements = {1, 2, 3}}");
}

void priority_queue_test() {
  std::priority_queue<int> i_am_empty;
  ComparePrettyPrintToChars(i_am_empty,
      "std::priority_queue wrapping = {std::vector of length 0, capacity 0}");

  std::priority_queue<int> one_two_three;
  one_two_three.push(11111);
  one_two_three.push(22222);
  one_two_three.push(33333);

  ComparePrettyPrintToRegex(one_two_three,
      R"(std::priority_queue wrapping = )"
      R"({std::vector of length 3, capacity 3 = {33333)");

  ComparePrettyPrintToRegex(one_two_three, ".*11111.*");
  ComparePrettyPrintToRegex(one_two_three, ".*22222.*");
}

void set_test() {
  std::set<int> i_am_empty;
  ComparePrettyPrintToChars(i_am_empty, "std::set is empty");

  std::set<int> one_two_three {3, 1, 2};
  ComparePrettyPrintToChars(one_two_three,
      "std::set with 3 elements = {1, 2, 3}");

  std::set<std::pair<int, int>> prime_pairs {
      std::make_pair(3, 5), std::make_pair(5, 7), std::make_pair(3, 5)};

  ComparePrettyPrintToChars(prime_pairs,
      "std::set with 2 elements = {"
      "{first = 3, second = 5}, {first = 5, second = 7}}");

  using using_set = std::set<int>;
  using_set other{1, 2, 3};
  ComparePrettyPrintToChars(other, "std::set with 3 elements = {1, 2, 3}");
}

void stack_test() {
  std::stack<int> test0;
  ComparePrettyPrintToChars(test0,
                            "std::stack wrapping = {std::deque is empty}");
  test0.push(5);
  test0.push(6);
  ComparePrettyPrintToChars(
      test0, "std::stack wrapping = {std::deque with 2 elements = {5, 6}}");
  std::stack<bool> test1;
  test1.push(true);
  test1.push(false);
  ComparePrettyPrintToChars(
      test1,
      "std::stack wrapping = {std::deque with 2 elements = {true, false}}");

  std::stack<std::string> test2;
  test2.push("Hello");
  test2.push("World");
  ComparePrettyPrintToChars(test2,
                            "std::stack wrapping = {std::deque with 2 elements "
                            "= {\"Hello\", \"World\"}}");
}

void multiset_test() {
  std::multiset<int> i_am_empty;
  ComparePrettyPrintToChars(i_am_empty, "std::multiset is empty");

  std::multiset<std::string> one_two_three {"1:one", "2:two", "3:three", "1:one"};
  ComparePrettyPrintToChars(one_two_three,
      "std::multiset with 4 elements = {"
      R"("1:one", "1:one", "2:two", "3:three"})");
}

void vector_test() {
  std::vector<bool> test0 = {true, false};
  ComparePrettyPrintToRegex(test0,
                            "std::vector<bool> of "
                            "length 2, capacity (32|64) = {1, 0}");
  for (int i = 0; i < 31; ++i) {
    test0.push_back(true);
    test0.push_back(false);
  }
  ComparePrettyPrintToRegex(
      test0,
      "std::vector<bool> of length 64, "
      "capacity 64 = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, "
      "0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, "
      "0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}");
  test0.push_back(true);
  ComparePrettyPrintToRegex(
      test0,
      "std::vector<bool> of length 65, "
      "capacity (96|128) = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, "
      "0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, "
      "0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}");

  std::vector<int> test1;
  ComparePrettyPrintToChars(test1, "std::vector of length 0, capacity 0");

  std::vector<int> test2 = {5, 6, 7};
  ComparePrettyPrintToChars(test2,
                            "std::vector of length "
                            "3, capacity 3 = {5, 6, 7}");

  std::vector<int, UncompressibleAllocator<int>> test3({7, 8});
  ComparePrettyPrintToChars(std::move(test3),
                            "std::vector of length "
                            "2, capacity 2 = {7, 8}");
}

void set_iterator_test() {
  std::set<int> one_two_three {1111, 2222, 3333};
  auto it = one_two_three.find(2222);
  MarkAsLive(it);
  CompareExpressionPrettyPrintToRegex("it",
      R"(std::__tree_const_iterator  = {\[0x[a-f0-9]+\] = 2222})");

  auto not_found = one_two_three.find(1234);
  MarkAsLive(not_found);
  // Because the end_node is not easily detected, just be sure it doesn't crash.
  CompareExpressionPrettyPrintToRegex("not_found",
      R"(std::__tree_const_iterator ( = {\[0x[a-f0-9]+\] = .*}|<error reading variable:.*>))");
}

void map_iterator_test() {
  std::map<int, std::string> one_two_three;
  one_two_three.insert({1, "one"});
  one_two_three.insert({2, "two"});
  one_two_three.insert({3, "three"});
  auto it = one_two_three.begin();
  MarkAsLive(it);
  CompareExpressionPrettyPrintToRegex("it",
      R"(std::__map_iterator  = )"
      R"({\[0x[a-f0-9]+\] = {first = 1, second = "one"}})");

  auto not_found = one_two_three.find(7);
  MarkAsLive(not_found);
  // Because the end_node is not easily detected, just be sure it doesn't crash.
  CompareExpressionPrettyPrintToRegex(
      "not_found", R"(std::__map_iterator ( = {\[0x[a-f0-9]+\] = .*}|<error reading variable:.*>))");
}

void unordered_set_test() {
  std::unordered_set<int> i_am_empty;
  ComparePrettyPrintToChars(i_am_empty, "std::unordered_set is empty");

  std::unordered_set<int> numbers {12345, 67890, 222333, 12345};
  numbers.erase(numbers.find(222333));
  ComparePrettyPrintToRegex(numbers, "std::unordered_set with 2 elements = ");
  ComparePrettyPrintToRegex(numbers, ".*12345.*");
  ComparePrettyPrintToRegex(numbers, ".*67890.*");

  std::unordered_set<std::string> colors {"red", "blue", "green"};
  ComparePrettyPrintToRegex(colors, "std::unordered_set with 3 elements = ");
  ComparePrettyPrintToRegex(colors, R"(.*"red".*)");
  ComparePrettyPrintToRegex(colors, R"(.*"blue".*)");
  ComparePrettyPrintToRegex(colors, R"(.*"green".*)");
}

void unordered_multiset_test() {
  std::unordered_multiset<int> i_am_empty;
  ComparePrettyPrintToChars(i_am_empty, "std::unordered_multiset is empty");

  std::unordered_multiset<int> numbers {12345, 67890, 222333, 12345};
  ComparePrettyPrintToRegex(numbers,
                            "std::unordered_multiset with 4 elements = ");
  ComparePrettyPrintToRegex(numbers, ".*12345.*12345.*");
  ComparePrettyPrintToRegex(numbers, ".*67890.*");
  ComparePrettyPrintToRegex(numbers, ".*222333.*");

  std::unordered_multiset<std::string> colors {"red", "blue", "green", "red"};
  ComparePrettyPrintToRegex(colors,
                            "std::unordered_multiset with 4 elements = ");
  ComparePrettyPrintToRegex(colors, R"(.*"red".*"red".*)");
  ComparePrettyPrintToRegex(colors, R"(.*"blue".*)");
  ComparePrettyPrintToRegex(colors, R"(.*"green".*)");
}

void unordered_map_test() {
  std::unordered_map<int, int> i_am_empty;
  ComparePrettyPrintToChars(i_am_empty, "std::unordered_map is empty");

  std::unordered_map<int, std::string> one_two_three;
  one_two_three.insert({1, "one"});
  one_two_three.insert({2, "two"});
  one_two_three.insert({3, "three"});
  ComparePrettyPrintToRegex(one_two_three,
                            "std::unordered_map with 3 elements = ");
  ComparePrettyPrintToRegex(one_two_three, R"(.*\[1\] = "one".*)");
  ComparePrettyPrintToRegex(one_two_three, R"(.*\[2\] = "two".*)");
  ComparePrettyPrintToRegex(one_two_three, R"(.*\[3\] = "three".*)");
}

void unordered_multimap_test() {
  std::unordered_multimap<int, int> i_am_empty;
  ComparePrettyPrintToChars(i_am_empty, "std::unordered_multimap is empty");

  std::unordered_multimap<int, std::string> one_two_three;
  one_two_three.insert({1, "one"});
  one_two_three.insert({2, "two"});
  one_two_three.insert({3, "three"});
  one_two_three.insert({2, "two"});
  ComparePrettyPrintToRegex(one_two_three,
                            "std::unordered_multimap with 4 elements = ");
  ComparePrettyPrintToRegex(one_two_three, R"(.*\[1\] = "one".*)");
  ComparePrettyPrintToRegex(one_two_three, R"(.*\[2\] = "two".*\[2\] = "two")");
  ComparePrettyPrintToRegex(one_two_three, R"(.*\[3\] = "three".*)");
}

void unordered_map_iterator_test() {
  std::unordered_map<int, int> ones_to_eights;
  ones_to_eights.insert({1, 8});
  ones_to_eights.insert({11, 88});
  ones_to_eights.insert({111, 888});

  auto ones_to_eights_begin = ones_to_eights.begin();
  MarkAsLive(ones_to_eights_begin);
  CompareExpressionPrettyPrintToRegex("ones_to_eights_begin",
      R"(std::__hash_map_iterator  = {\[1+\] = 8+})");

  auto not_found = ones_to_eights.find(5);
  MarkAsLive(not_found);
  CompareExpressionPrettyPrintToRegex("not_found",
      R"(std::__hash_map_iterator = end\(\))");
}

void unordered_set_iterator_test() {
  std::unordered_set<int> ones;
  ones.insert(111);
  ones.insert(1111);
  ones.insert(11111);

  auto ones_begin = ones.begin();
  MarkAsLive(ones_begin);
  CompareExpressionPrettyPrintToRegex("ones_begin",
      R"(std::__hash_const_iterator  = {1+})");

  auto not_found = ones.find(5);
  MarkAsLive(not_found);
  CompareExpressionPrettyPrintToRegex("not_found",
      R"(std::__hash_const_iterator = end\(\))");
}

// Check that libc++ pretty printers do not handle pointers.
void pointer_negative_test() {
  int abc = 123;
  int *int_ptr = &abc;
  // Check that the result is equivalent to "p/r int_ptr" command.
  ComparePrettyPrintToRegex(int_ptr, R"(\(int \*\) 0x[a-f0-9]+)");
}

void shared_ptr_test() {
  // Shared ptr tests while using test framework call another function
  // due to which there is one more count for the pointer. Hence, all the
  // following tests are testing with expected count plus 1.
  std::shared_ptr<const int> test0 = std::make_shared<const int>(5);
  // The python regular expression matcher treats newlines as significant, so
  // these regular expressions should be on one line.
  ComparePrettyPrintToRegex(
      test0,
      R"(std::shared_ptr<int> count [2\?], weak [0\?]( \(libc\+\+ missing debug info\))? containing = {__ptr_ = 0x[a-f0-9]+})");

  std::shared_ptr<const int> test1(test0);
  ComparePrettyPrintToRegex(
      test1,
      R"(std::shared_ptr<int> count [3\?], weak [0\?]( \(libc\+\+ missing debug info\))? containing = {__ptr_ = 0x[a-f0-9]+})");

  {
    std::weak_ptr<const int> test2 = test1;
    ComparePrettyPrintToRegex(
        test0,
        R"(std::shared_ptr<int> count [3\?], weak [1\?]( \(libc\+\+ missing debug info\))? containing = {__ptr_ = 0x[a-f0-9]+})");
  }

  ComparePrettyPrintToRegex(
      test0,
      R"(std::shared_ptr<int> count [3\?], weak [0\?]( \(libc\+\+ missing debug info\))? containing = {__ptr_ = 0x[a-f0-9]+})");

  std::shared_ptr<const int> test3;
  ComparePrettyPrintToChars(test3, "std::shared_ptr is nullptr");
}

void streampos_test() {
  std::streampos test0 = 67;
  ComparePrettyPrintToChars(
      test0, "std::fpos with stream offset:67 with state: {count:0 value:0}");
  std::istringstream input("testing the input stream here");
  std::streampos test1 = input.tellg();
  ComparePrettyPrintToChars(
      test1, "std::fpos with stream offset:0 with state: {count:0 value:0}");
  std::unique_ptr<char[]> buffer(new char[5]);
  input.read(buffer.get(), 5);
  test1 = input.tellg();
  ComparePrettyPrintToChars(
      test1, "std::fpos with stream offset:5 with state: {count:0 value:0}");
}

int main(int, char**) {
  framework_self_test();

  string_test();
  a_namespace::string_view_test();

  //u16string_test();
  u32string_test();
  tuple_test();
  unique_ptr_test();
  shared_ptr_test();
  bitset_test();
  list_test();
  deque_test();
  map_test();
  multimap_test();
  queue_test();
  priority_queue_test();
  stack_test();
  set_test();
  multiset_test();
  vector_test();
  set_iterator_test();
  map_iterator_test();
  unordered_set_test();
  unordered_multiset_test();
  unordered_map_test();
  unordered_multimap_test();
  unordered_map_iterator_test();
  unordered_set_iterator_test();
  pointer_negative_test();
  streampos_test();
  return 0;
}
