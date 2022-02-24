// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -Wno-strict-prototypes -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-windows -fsyntax-only -Wno-strict-prototypes -verify %s

#define SWIFTCALL __attribute__((swiftcall))
#define SWIFTASYNCCALL __attribute__((swiftasynccall))
#define INDIRECT_RESULT __attribute__((swift_indirect_result))
#define ERROR_RESULT __attribute__((swift_error_result))
#define CONTEXT __attribute__((swift_context))
#define ASYNC_CONTEXT __attribute__((swift_async_context))

int notAFunction SWIFTCALL; // expected-warning {{'swiftcall' only applies to function types; type here is 'int'}}
int notAnAsyncFunction SWIFTASYNCCALL; // expected-warning {{'swiftasynccall' only applies to function types; type here is 'int'}}
void variadic(int x, ...) SWIFTCALL; // expected-error {{variadic function cannot use swiftcall calling convention}}
void variadic_async(int x, ...) SWIFTASYNCCALL; // expected-error {{variadic function cannot use swiftasynccall calling convention}}
void unprototyped() SWIFTCALL; // expected-error {{function with no prototype cannot use the swiftcall calling convention}}
void unprototyped_async() SWIFTASYNCCALL; // expected-error {{function with no prototype cannot use the swiftasynccall calling convention}}
void multiple_ccs(int x) SWIFTCALL __attribute__((vectorcall)); // expected-error {{vectorcall and swiftcall attributes are not compatible}}
void multiple_ccs_async(int x) SWIFTASYNCCALL __attribute__((vectorcall)); // expected-error {{vectorcall and swiftasynccall attributes are not compatible}}
void (*functionPointer)(void) SWIFTCALL;
void (*asyncFunctionPointer)(void) SWIFTASYNCCALL;

void indirect_result_nonswift(INDIRECT_RESULT void *out); // expected-error {{'swift_indirect_result' parameter can only be used with swiftcall or swiftasynccall calling convention}}
void indirect_result_bad_position(int first, INDIRECT_RESULT void *out) SWIFTCALL; // expected-error {{'swift_indirect_result' parameters must be first parameters of function}}
void indirect_result_bad_type(INDIRECT_RESULT int out) SWIFTCALL; // expected-error {{'swift_indirect_result' parameter must have pointer type; type here is 'int'}}
void indirect_result_single(INDIRECT_RESULT void *out) SWIFTCALL;
void indirect_result_multiple(INDIRECT_RESULT void *out1, INDIRECT_RESULT void *out2) SWIFTCALL;
void indirect_result_single_async(INDIRECT_RESULT void *out) SWIFTASYNCCALL;
void indirect_result_multiple_async(INDIRECT_RESULT void *out1, INDIRECT_RESULT void *out2) SWIFTASYNCCALL;

void error_result_nonswift(ERROR_RESULT void **error); // expected-error {{'swift_error_result' parameter can only be used with swiftcall calling convention}} expected-error{{'swift_error_result' parameter must follow 'swift_context' parameter}}
void error_result_bad_position2(int first, ERROR_RESULT void **error) SWIFTCALL; // expected-error {{'swift_error_result' parameter must follow 'swift_context' parameter}}
void error_result_bad_type(CONTEXT void *context, ERROR_RESULT int error) SWIFTCALL; // expected-error {{'swift_error_result' parameter must have pointer to unqualified pointer type; type here is 'int'}}
void error_result_bad_type2(CONTEXT void *context, ERROR_RESULT int *error) SWIFTCALL; // expected-error {{'swift_error_result' parameter must have pointer to unqualified pointer type; type here is 'int *'}}
void error_result_okay(int a, int b, CONTEXT void *context, ERROR_RESULT void **error) SWIFTCALL;
void error_result_okay2(CONTEXT void *context, ERROR_RESULT void **error, void *selfType, char **selfWitnessTable) SWIFTCALL;

void context_nonswift(CONTEXT void *context); // expected-error {{'swift_context' parameter can only be used with swiftcall or swiftasynccall calling convention}}
void context_bad_type(CONTEXT int context) SWIFTCALL; // expected-error {{'swift_context' parameter must have pointer type; type here is 'int'}}
void context_okay(CONTEXT void *context) SWIFTCALL;
void context_okay2(CONTEXT void *context, void *selfType, char **selfWitnessTable) SWIFTCALL;
void context_okay_async(CONTEXT void *context) SWIFTASYNCCALL;
void context_okay2_async(CONTEXT void *context, void *selfType, char **selfWitnessTable) SWIFTASYNCCALL;

void async_context_okay_for_now(ASYNC_CONTEXT void *context);
void async_context_bad_type(ASYNC_CONTEXT int context) SWIFTCALL; // expected-error {{'swift_async_context' parameter must have pointer type; type here is 'int'}}
void async_context_bad_pos(int context) ASYNC_CONTEXT SWIFTCALL; // expected-warning {{'swift_async_context' attribute only applies to parameters}}
void async_context_bad_args(__attribute__((swift_async_context(1))) void *context) SWIFTCALL; // expected-error {{'swift_async_context' attribute takes no arguments}}
void async_context_okay(ASYNC_CONTEXT void *context) SWIFTCALL;
void async_context_okay2(ASYNC_CONTEXT void *context, void *selfType, char **selfWitnessTable) SWIFTCALL;
