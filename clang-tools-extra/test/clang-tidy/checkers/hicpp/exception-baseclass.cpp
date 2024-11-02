// RUN: %check_clang_tidy %s hicpp-exception-baseclass %t -- -- -fcxx-exceptions

namespace std {
class exception {};
class invalid_argument : public exception {};
} // namespace std

class derived_exception : public std::exception {};
class deep_hierarchy : public derived_exception {};
class non_derived_exception {};
class terrible_idea : public non_derived_exception, public derived_exception {};

// FIXME: More complicated kinds of inheritance should be checked later, but there is
// currently no way use ASTMatchers for this kind of task.
#if 0
class bad_inheritance : private std::exception {};
class no_good_inheritance : protected std::exception {};
class really_creative : public non_derived_exception, private std::exception {};
#endif

void problematic() {
  try {
    throw int(42);
    // CHECK-NOTES: [[@LINE-1]]:11: warning: throwing an exception whose type 'int' is not derived from 'std::exception'
  } catch (int e) {
  }
  throw int(42);
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'int' is not derived from 'std::exception'

  try {
    throw 12;
    // CHECK-NOTES: [[@LINE-1]]:11: warning: throwing an exception whose type 'int' is not derived from 'std::exception'
  } catch (...) {
    throw; // Ok, even if the type is not known, conforming code can never rethrow a non-std::exception object.
  }

  try {
    throw non_derived_exception();
    // CHECK-NOTES: [[@LINE-1]]:11: warning: throwing an exception whose type 'non_derived_exception' is not derived from 'std::exception'
    // CHECK-NOTES: 10:1: note: type defined here
  } catch (non_derived_exception &e) {
  }
  throw non_derived_exception();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'non_derived_exception' is not derived from 'std::exception'
  // CHECK-NOTES: 10:1: note: type defined here

// FIXME: More complicated kinds of inheritance should be checked later, but there is
// currently no way use ASTMatchers for this kind of task.
#if 0
  // Handle private inheritance cases correctly.
  try {
    throw bad_inheritance();
    // CHECK NOTES: [[@LINE-1]]:11: warning: throwing an exception whose type 'bad_inheritance' is not derived from 'std::exception'
    // CHECK NOTES: 11:1: note: type defined here
    throw no_good_inheritance();
    // CHECK NOTES: [[@LINE-1]]:11: warning: throwing an exception whose type 'no_good_inheritance' is not derived from 'std::exception'
    // CHECK NOTES: 12:1: note: type defined here
    throw really_creative();
    // CHECK NOTES: [[@LINE-1]]:11: warning: throwing an exception whose type 'really_creative' is not derived from 'std::exception'
    // CHECK NOTES: 13:1: note: type defined here
  } catch (...) {
  }
  throw bad_inheritance();
  // CHECK NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'bad_inheritance' is not derived from 'std::exception'
  // CHECK NOTES: 11:1: note: type defined here
  throw no_good_inheritance();
  // CHECK NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'no_good_inheritance' is not derived from 'std::exception'
  // CHECK NOTES: 12:1: note: type defined here
  throw really_creative();
  // CHECK NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'really_creative' is not derived from 'std::exception'
  // CHECK NOTES: 13:1: note: type defined here
#endif
}

void allowed_throws() {
  try {
    throw std::exception();     // Ok
  } catch (std::exception &e) { // Ok
  }
  throw std::exception();

  try {
    throw derived_exception();     // Ok
  } catch (derived_exception &e) { // Ok
  }
  throw derived_exception(); // Ok

  try {
    throw deep_hierarchy();     // Ok, multiple levels of inheritance
  } catch (deep_hierarchy &e) { // Ok
  }
  throw deep_hierarchy(); // Ok

  try {
    throw terrible_idea();      // Ok, but multiple inheritance isn't clean
  } catch (std::exception &e) { // Can be caught as std::exception, even with multiple inheritance
  }
  throw terrible_idea(); // Ok, but multiple inheritance
}

void test_lambdas() {
  auto BadLambda = []() { throw int(42); };
  // CHECK-NOTES: [[@LINE-1]]:33: warning: throwing an exception whose type 'int' is not derived from 'std::exception'
  auto GoodLambda = []() { throw derived_exception(); };
}

// Templated function that throws exception based on template type
template <typename T>
void ThrowException() { throw T(); }
// CHECK-NOTES: [[@LINE-1]]:31: warning: throwing an exception whose type 'bad_generic_exception<int>' is not derived from 'std::exception'
// CHECK-NOTES: [[@LINE-2]]:31: note: type 'bad_generic_exception<int>' is a template instantiation of 'T'
// CHECK-NOTES: [[@LINE+25]]:1: note: type defined here

// CHECK-NOTES: [[@LINE-5]]:31: warning: throwing an exception whose type 'bad_generic_exception<std::exception>' is not derived from 'std::exception'
// CHECK-NOTES: [[@LINE-6]]:31: note: type 'bad_generic_exception<std::exception>' is a template instantiation of 'T'
// CHECK-NOTES: [[@LINE+21]]:1: note: type defined here

// CHECK-NOTES: [[@LINE-9]]:31: warning: throwing an exception whose type 'exotic_exception<non_derived_exception>' is not derived from 'std::exception'
// CHECK-NOTES: [[@LINE-10]]:31: note: type 'exotic_exception<non_derived_exception>' is a template instantiation of 'T'
// CHECK-NOTES: [[@LINE+20]]:1: note: type defined here

// CHECK-NOTES: [[@LINE-13]]:31: warning: throwing an exception whose type 'int' is not derived from 'std::exception'
// CHECK-NOTES: [[@LINE-14]]:31: note: type 'int' is a template instantiation of 'T'

// CHECK-NOTES: [[@LINE-16]]:31: warning: throwing an exception whose type 'non_derived_exception' is not derived from 'std::exception'
// CHECK-NOTES: [[@LINE-17]]:31: note: type 'non_derived_exception' is a template instantiation of 'T'
// CHECK-NOTES: 10:1: note: type defined here

#define THROW_EXCEPTION(CLASS) ThrowException<CLASS>()
#define THROW_BAD_EXCEPTION throw int(42);
#define THROW_GOOD_EXCEPTION throw std::exception();
#define THROW_DERIVED_EXCEPTION throw deep_hierarchy();

template <typename T>
class generic_exception : std::exception {};

template <typename T>
class bad_generic_exception {};

template <typename T>
class exotic_exception : public T {};

void generic_exceptions() {
  THROW_EXCEPTION(int);
  THROW_EXCEPTION(non_derived_exception);
  THROW_EXCEPTION(std::exception);    // Ok
  THROW_EXCEPTION(derived_exception); // Ok
  THROW_EXCEPTION(deep_hierarchy);    // Ok

  THROW_BAD_EXCEPTION;
  // CHECK-NOTES: [[@LINE-1]]:3: warning: throwing an exception whose type 'int' is not derived from 'std::exception'
  // CHECK-NOTES: [[@LINE-22]]:35: note: expanded from macro 'THROW_BAD_EXCEPTION'
  THROW_GOOD_EXCEPTION;
  THROW_DERIVED_EXCEPTION;

  throw generic_exception<int>();            // Ok,
  THROW_EXCEPTION(generic_exception<float>); // Ok

  throw bad_generic_exception<int>();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'bad_generic_exception<int>' is not derived from 'std::exception'
  // CHECK-NOTES: [[@LINE-24]]:1: note: type defined here
  throw bad_generic_exception<std::exception>();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'bad_generic_exception<std::exception>' is not derived from 'std::exception'
  // CHECK-NOTES: [[@LINE-27]]:1: note: type defined here
  THROW_EXCEPTION(bad_generic_exception<int>);
  THROW_EXCEPTION(bad_generic_exception<std::exception>);

  throw exotic_exception<non_derived_exception>();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'exotic_exception<non_derived_exception>' is not derived from 'std::exception'
  // CHECK-NOTES: [[@LINE-30]]:1: note: type defined here
  THROW_EXCEPTION(exotic_exception<non_derived_exception>);

  throw exotic_exception<derived_exception>();          // Ok
  THROW_EXCEPTION(exotic_exception<derived_exception>); // Ok
}

// Test for typedefed exception types
typedef int TypedefedBad;
typedef derived_exception TypedefedGood;
using UsingBad = int;
using UsingGood = deep_hierarchy;

void typedefed() {
  throw TypedefedBad();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'TypedefedBad' (aka 'int') is not derived from 'std::exception'
  // CHECK-NOTES: [[@LINE-8]]:1: note: type defined here
  throw TypedefedGood(); // Ok

  throw UsingBad();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'UsingBad' (aka 'int') is not derived from 'std::exception'
  // CHECK-NOTES: [[@LINE-11]]:1: note: type defined here
  throw UsingGood(); // Ok
}

// Fix PR37913
struct invalid_argument_maker {
  ::std::invalid_argument operator()() const;
};
struct int_maker {
  int operator()() const;
};

template <typename T>
void templated_thrower() {
  throw T{}();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'int' is not derived from 'std::exception'
}
template <typename T>
void templated_thrower2() {
  T ExceptionFactory; // This test found a <dependant-type> which did not happend with 'throw T{}()'
  throw ExceptionFactory();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'int' is not derived from 'std::exception'
}

void exception_created_with_function() {
  templated_thrower<invalid_argument_maker>();
  templated_thrower<int_maker>();

  templated_thrower2<invalid_argument_maker>();
  templated_thrower2<int_maker>();

  throw invalid_argument_maker{}();
  throw int_maker{}();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'int' is not derived from 'std::exception'
}

struct invalid_argument_factory {
  ::std::invalid_argument make_exception() const;
};

struct int_factory {
  int make_exception() const;
};

template <typename T>
void templated_factory() {
  T f;
  throw f.make_exception();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'int' is not derived from 'std::exception'
}
template <typename T>
void templated_factory2() {
  throw T().make_exception();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'int' is not derived from 'std::exception'
}

void exception_from_factory() {
  templated_factory<invalid_argument_factory>();
  templated_factory<int_factory>();

  templated_factory2<invalid_argument_factory>();
  templated_factory2<int_factory>();

  throw invalid_argument_factory().make_exception();
  throw int_factory().make_exception();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'int' is not derived from 'std::exception'

  invalid_argument_factory inv_f;
  throw inv_f.make_exception();

  int_factory int_f;
  throw int_f.make_exception();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'int' is not derived from 'std::exception'
}

template <typename T>
struct ThrowClassTemplateParam {
  ThrowClassTemplateParam() { throw T(); }
  // CHECK-NOTES: [[@LINE-1]]:37: warning: throwing an exception whose type 'int' is not derived from 'std::exception'
  // CHECK-NOTES: [[@LINE-2]]:37: note: type 'int' is a template instantiation of 'T'
};

template <int V>
struct ThrowValueTemplate {
  ThrowValueTemplate() { throw V; }
  // CHECK-NOTES: [[@LINE-1]]:32: warning: throwing an exception whose type 'int' is not derived from 'std::exception'
};

void class_templates() {
  ThrowClassTemplateParam<int> IntThrow;
  ThrowClassTemplateParam<std::invalid_argument> ArgThrow;

  ThrowValueTemplate<42> ValueThrow;
}
