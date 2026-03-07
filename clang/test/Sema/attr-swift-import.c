// RUN: %clang_cc1 -fblocks -fsyntax-only -std=c23 -verify -Wunknown-attributes %s

#pragma mark - swift_async_name

[[clang::swift_async_name("testAsyncFunc()")]]
void test_async_func(void (^completion)(void));

// expected-warning@+1 {{too many parameters in the signature specified by the 'clang::swift_async_name' attribute (expected 0; got 1)}}
[[clang::swift_async_name("testAsyncFuncWithoutCompletionHandler(x:)")]]
void test_async_func_with_no_completion_handler(int x);

// expected-warning@+1 {{'clang::swift_async_name' attribute cannot be applied to a function with no parameters}}
[[clang::swift_async_name("testAsyncFuncWithNoParameters()")]]
void test_async_func_with_no_parameters(void);

[[clang::swift_async_name("testAsyncFuncWithParameter(x:)")]]
void test_async_func_with_parameter(int x, void (^completion)(void));

// expected-warning@+1 {{too many parameters in the signature specified by the 'clang::swift_async_name' attribute (expected 1; got 2)}}
[[clang::swift_async_name("testAsyncFuncWithTooManyParameters(x:completion:)")]]
void test_async_func_with_too_many_parameters(int x, void (^completion)(void));

#pragma mark - swift_attr

[[clang::swift_attr("Escapable")]]
typedef struct test_escapable_t test_escapable_t;

#pragma mark - swift_name

[[clang::swift_name("TestSwiftName")]]
typedef struct test_swift_name_s *test_swift_name_t;

[[clang::swift_name("TestSwiftName.init()")]]
test_swift_name_t test_swift_name_init(void);

[[clang::swift_name("TestSwiftName.mutatingMethod(self:)")]]
void test_swift_name_mutating_method(test_swift_name_t test_swift_name);

[[clang::swift_name("getter:TestSwiftName.property(self:)")]]
int test_swift_name_get_property(const test_swift_name_t test_swift_name);

[[clang::swift_name("setter:TestSwiftName.property(self:newValue:)")]]
int test_swift_name_set_property(test_swift_name_t test_swift_name, int x);

enum [[clang::swift_name("TestSwiftNameEnum")]] test_swift_name_e {
  test_swift_name_a [[clang::swift_name("a")]] = 1,
  test_swift_name_b
};

// expected-error@+1 {{'clang::swift_name' attribute cannot be applied to types}}
void test_swift_name_func_applied_to_type(int x) [[clang::swift_name("f(_:)")]];

#pragma mark - swift_newtype, swift_wrapper

[[clang::swift_newtype(enum)]]
typedef struct test_newtype_e test_newtype_t;

// expected-error@+1 {{'clang::swift_newtype' attribute only applies to typedefs}}
struct [[clang::swift_newtype(struct)]] test_newtype_s;

[[clang::swift_wrapper(struct)]]
typedef struct test_wrapper_s test_wrapper_t;

#pragma mark - swift_private

enum [[clang::swift_private]] test_swift_private_e {
  test_swift_private_a,
  test_swift_private_b
};

struct [[clang::swift_private]] test_swift_private_s;

[[clang::swift_private]]
typedef struct test_swift_private_s test_swift_private_t;

[[clang::swift_private]]
void test_swift_private_func(void);
