// RUN: %check_clang_tidy %s modernize-nlohmann-json-explicit-conversions %t -- -- -isystem %clang_tidy_headers

#include <string>

namespace nlohmann
{
  class basic_json
  {
  public:
    template <typename ValueType>
    ValueType get() const
    {
      return ValueType{};
    }

    // nlohmann::json uses SFINAE to limit the types that can be converted to.
    // Rather than do that here, let's just provide the overloads we need
    // instead.
    operator int() const
    {
      return get<int>();
    }

    operator double() const
    {
      return get<double>();
    }

    operator std::string() const
    {
      return get<std::string>();
    }

    int otherMember() const;
  };

  class iterator
  {
  public:
    basic_json &operator*();
    basic_json *operator->();
  };

  using json = basic_json;
}

using nlohmann::json;
using nlohmann::iterator;

int get_int(json &j)
{
  return j;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: implicit nlohmann::json conversion to int should be explicit [modernize-nlohmann-json-explicit-conversions]
  // CHECK-FIXES: return j.get<int>();
}

std::string get_string(json &j)
{
  return j;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: implicit nlohmann::json conversion to std::string should be explicit [modernize-nlohmann-json-explicit-conversions]
  // CHECK-FIXES: return j.get<std::string>();
}

int get_int_ptr(json *j)
{
  return *j;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: implicit nlohmann::json conversion to int should be explicit [modernize-nlohmann-json-explicit-conversions]
  // CHECK-FIXES: return j->get<int>();
}

int get_int_ptr_expr(json *j)
{
  return *(j+1);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: implicit nlohmann::json conversion to int should be explicit [modernize-nlohmann-json-explicit-conversions]
  // CHECK-FIXES: return (j+1)->get<int>();
}

int get_int_iterator(iterator i)
{
  return *i;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: implicit nlohmann::json conversion to int should be explicit [modernize-nlohmann-json-explicit-conversions]
  // CHECK-FIXES: return i->get<int>();
}

int get_int_fn()
{
  extern json get_json();
  return get_json();
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: implicit nlohmann::json conversion to int should be explicit [modernize-nlohmann-json-explicit-conversions]
  // CHECK-FIXES: return get_json().get<int>();
}

double get_double_fn_ref()
{
  extern nlohmann::json &get_json_ref();
  return get_json_ref();
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: implicit nlohmann::json conversion to double should be explicit [modernize-nlohmann-json-explicit-conversions]
  // CHECK-FIXES: return get_json_ref().get<double>();
}

std::string get_string_fn_ptr()
{
  extern json *get_json_ptr();
  return *get_json_ptr();
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: implicit nlohmann::json conversion to std::string should be explicit [modernize-nlohmann-json-explicit-conversions]
  // CHECK-FIXES: return get_json_ptr()->get<std::string>();
}

int call_other_member(nlohmann::json &j)
{
  return j.otherMember();
}

int call_other_member_ptr(nlohmann::json *j)
{
  return j->otherMember();
}

int call_other_member_iterator(iterator i)
{
  return i->otherMember();
}
