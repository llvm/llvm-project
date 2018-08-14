// RUN: %check_clang_tidy %s hicpp-exception-baseclass %t -- -- -fcxx-exceptions

namespace std {
class exception {};
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
    // CHECK-NOTES: 9:1: note: type defined here
  } catch (non_derived_exception &e) {
  }
  throw non_derived_exception();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'non_derived_exception' is not derived from 'std::exception'
  // CHECK-NOTES: 9:1: note: type defined here

// FIXME: More complicated kinds of inheritance should be checked later, but there is
// currently no way use ASTMatchers for this kind of task.
#if 0
  // Handle private inheritance cases correctly.
  try {
    throw bad_inheritance();
    // CHECK MESSAGES: [[@LINE-1]]:11: warning: throwing an exception whose type 'bad_inheritance' is not derived from 'std::exception'
    // CHECK MESSAGES: 10:1: note: type defined here
    throw no_good_inheritance();
    // CHECK MESSAGES: [[@LINE-1]]:11: warning: throwing an exception whose type 'no_good_inheritance' is not derived from 'std::exception'
    // CHECK MESSAGES: 11:1: note: type defined here
    throw really_creative();
    // CHECK MESSAGES: [[@LINE-1]]:11: warning: throwing an exception whose type 'really_creative' is not derived from 'std::exception'
    // CHECK MESSAGES: 12:1: note: type defined here
  } catch (...) {
  }
  throw bad_inheritance();
  // CHECK MESSAGES: [[@LINE-1]]:9: warning: throwing an exception whose type 'bad_inheritance' is not derived from 'std::exception'
  // CHECK MESSAGES: 10:1: note: type defined here
  throw no_good_inheritance();
  // CHECK MESSAGES: [[@LINE-1]]:9: warning: throwing an exception whose type 'no_good_inheritance' is not derived from 'std::exception'
  // CHECK MESSAGES: 11:1: note: type defined here
  throw really_creative();
  // CHECK MESSAGES: [[@LINE-1]]:9: warning: throwing an exception whose type 'really_creative' is not derived from 'std::exception'
  // CHECK MESSAGES: 12:1: note: type defined here
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
    throw terrible_idea();     // Ok, but multiple inheritance isn't clean
  } catch (std::exception &e) { // Can be caught as std::exception, even with multiple inheritance
  }
  throw terrible_idea(); // Ok, but multiple inheritance
}

// Templated function that throws exception based on template type
template <typename T>
void ThrowException() { throw T(); }
// CHECK-NOTES: [[@LINE-1]]:31: warning: throwing an exception whose type 'bad_generic_exception<int>' is not derived from 'std::exception'
// CHECK-NOTES: 120:1: note: type defined here
// CHECK-NOTES: [[@LINE-3]]:31: warning: throwing an exception whose type 'bad_generic_exception<std::exception>' is not derived from 'std::exception'
// CHECK-NOTES: 120:1: note: type defined here
// CHECK-NOTES: [[@LINE-5]]:31: warning: throwing an exception whose type 'exotic_exception<non_derived_exception>' is not derived from 'std::exception'
// CHECK-NOTES: 123:1: note: type defined here
// CHECK-NOTES: [[@LINE-7]]:31: warning: throwing an exception whose type 'int' is not derived from 'std::exception'
// CHECK-NOTES: [[@LINE-8]]:31: warning: throwing an exception whose type 'non_derived_exception' is not derived from 'std::exception'
// CHECK-NOTES: 9:1: note: type defined here
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
  // CHECK MESSAGES: [[@LINE-1]]:3: warning: throwing an exception whose type 'int' is not derived from 'std::exception'
  THROW_EXCEPTION(non_derived_exception);
  // CHECK MESSAGES: [[@LINE-1]]:3: warning: throwing an exception whose type 'non_derived_exception' is not derived from 'std::exception'
  // CHECK MESSAGES: 9:1: note: type defined here
  THROW_EXCEPTION(std::exception);    // Ok
  THROW_EXCEPTION(derived_exception); // Ok
  THROW_EXCEPTION(deep_hierarchy);    // Ok

  THROW_BAD_EXCEPTION;
  // CHECK-NOTES: [[@LINE-1]]:3: warning: throwing an exception whose type 'int' is not derived from 'std::exception'
  // CHECK-NOTES: [[@LINE-25]]:35: note: expanded from macro 'THROW_BAD_EXCEPTION'
  THROW_GOOD_EXCEPTION;
  THROW_DERIVED_EXCEPTION;

  throw generic_exception<int>();            // Ok,
  THROW_EXCEPTION(generic_exception<float>); // Ok

  throw bad_generic_exception<int>();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'bad_generic_exception<int>' is not derived from 'std::exception'
  // CHECK-NOTES: 120:1: note: type defined here
  throw bad_generic_exception<std::exception>();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'bad_generic_exception<std::exception>' is not derived from 'std::exception'
  // CHECK-NOTES: 120:1: note: type defined here
  THROW_EXCEPTION(bad_generic_exception<int>);
  // CHECK MESSAGES: [[@LINE-1]]:3: warning: throwing an exception whose type 'bad_generic_exception<int>' is not derived from 'std::exception'
  THROW_EXCEPTION(bad_generic_exception<std::exception>);
  // CHECK MESSAGES: [[@LINE-1]]:3: warning: throwing an exception whose type 'bad_generic_exception<std::exception>' is not derived from 'std::exception'

  throw exotic_exception<non_derived_exception>();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'exotic_exception<non_derived_exception>' is not derived from 'std::exception'
  // CHECK-NOTES: 123:1: note: type defined here
  THROW_EXCEPTION(exotic_exception<non_derived_exception>);
  // CHECK MESSAGES: [[@LINE-1]]:3: warning: throwing an exception whose type 'exotic_exception<non_derived_exception>' is not derived from 'std::exception'

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
  // CHECK-NOTES: 167:1: note: type defined here
  throw TypedefedGood(); // Ok

  throw UsingBad();
  // CHECK-NOTES: [[@LINE-1]]:9: warning: throwing an exception whose type 'UsingBad' (aka 'int') is not derived from 'std::exception'
  // CHECK-NOTES: 169:1: note: type defined here
  throw UsingGood(); // Ok
}
