// RUN: %clang_cc1 -fsyntax-only -Wthread-safety -verify %s

typedef int __attribute__((capability("role"))) ThreadRole;
struct __attribute__((shared_capability("mutex"))) Mutex {};
struct NotACapability {};

// Put capability attributes on unions
union __attribute__((capability("mutex"))) MutexUnion { int a; char* b; };
typedef union { int a; char* b; } __attribute__((capability("mutex"))) MutexUnion2;

// Test a different capability name
struct __attribute__((capability("custom"))) CustomName {};

int Test1 __attribute__((capability("test1")));  // expected-error {{'capability' attribute only applies to structs, unions, classes, and typedefs}}
int Test2 __attribute__((shared_capability("test2"))); // expected-error {{'shared_capability' attribute only applies to structs, unions, classes, and typedefs}}
int Test3 __attribute__((acquire_capability("test3")));  // expected-warning {{'acquire_capability' attribute only applies to functions}}
int Test4 __attribute__((try_acquire_capability("test4"))); // expected-error {{'try_acquire_capability' attribute only applies to functions}}
int Test5 __attribute__((release_capability("test5"))); // expected-warning {{'release_capability' attribute only applies to functions}}

struct __attribute__((capability(12))) Test3 {}; // expected-error {{expected string literal as argument of 'capability' attribute}}
struct __attribute__((shared_capability(Test2))) Test4 {}; // expected-error {{expected string literal as argument of 'shared_capability' attribute}}

struct __attribute__((capability)) Test5 {}; // expected-error {{'capability' attribute takes one argument}}
struct __attribute__((shared_capability("test1", 12))) Test6 {}; // expected-error {{'shared_capability' attribute takes one argument}}

struct NotACapability BadCapability;
ThreadRole GUI, Worker;
void Func1(void) __attribute__((requires_capability(GUI))) {}
void Func2(void) __attribute__((requires_shared_capability(Worker))) {}

void Func3(void) __attribute__((requires_capability)) {}  // expected-error {{'requires_capability' attribute takes at least 1 argument}}
void Func4(void) __attribute__((requires_shared_capability)) {}  // expected-error {{'requires_shared_capability' attribute takes at least 1 argument}}

void Func5(void) __attribute__((requires_capability(1))) {}  // expected-warning {{'requires_capability' attribute requires arguments whose type is annotated with 'capability' attribute; type here is 'int'}}
void Func6(void) __attribute__((requires_shared_capability(BadCapability))) {}  // expected-warning {{'requires_shared_capability' attribute requires arguments whose type is annotated with 'capability' attribute; type here is 'struct NotACapability'}}

void Func7(void) __attribute__((assert_capability(GUI))) {}
void Func8(void) __attribute__((assert_shared_capability(GUI))) {}

void Func9(void) __attribute__((assert_capability())) {} // expected-warning {{'assert_capability' attribute without capability arguments can only be applied to non-static methods of a class}}
void Func10(void) __attribute__((assert_shared_capability())) {} // expected-warning {{'assert_shared_capability' attribute without capability arguments can only be applied to non-static methods of a class}}

void Func11(void) __attribute__((acquire_capability(GUI))) {}
void Func12(void) __attribute__((acquire_shared_capability(GUI))) {}

void Func13(void) __attribute__((acquire_capability())) {} // expected-warning {{'acquire_capability' attribute without capability arguments can only be applied to non-static methods of a class}}
void Func14(void) __attribute__((acquire_shared_capability())) {} // expected-warning {{'acquire_shared_capability' attribute without capability arguments can only be applied to non-static methods of a class}}

void Func15(void) __attribute__((release_capability(GUI))) {}
void Func16(void) __attribute__((release_shared_capability(GUI))) {}
void Func17(void) __attribute__((release_generic_capability(GUI))) {}

void Func18(void) __attribute__((release_capability())) {} // expected-warning {{'release_capability' attribute without capability arguments can only be applied to non-static methods of a class}}
void Func19(void) __attribute__((release_shared_capability())) {} // expected-warning {{'release_shared_capability' attribute without capability arguments can only be applied to non-static methods of a class}}
void Func20(void) __attribute__((release_generic_capability())) {} // expected-warning {{'release_generic_capability' attribute without capability arguments can only be applied to non-static methods of a class}}

int Func21(void) __attribute__((try_acquire_capability(1))) {} // expected-warning {{'try_acquire_capability' attribute without capability arguments can only be applied to non-static methods of a class}}
int Func22(void) __attribute__((try_acquire_shared_capability(1))) {} // expected-warning {{'try_acquire_shared_capability' attribute without capability arguments can only be applied to non-static methods of a class}}

int Func23(void) __attribute__((try_acquire_capability(1, GUI))) {}
int Func24(void) __attribute__((try_acquire_shared_capability(1, GUI))) {}

int Func25(void) __attribute__((try_acquire_capability())) {} // expected-error {{'try_acquire_capability' attribute takes at least 1 argument}}
int Func26(void) __attribute__((try_acquire_shared_capability())) {} // expected-error {{'try_acquire_shared_capability' attribute takes at least 1 argument}}

// Test that boolean logic works with capability attributes
void Func27(void) __attribute__((requires_capability(!GUI)));
void Func28(void) __attribute__((requires_capability(GUI && Worker)));
void Func29(void) __attribute__((requires_capability(GUI || Worker)));
void Func30(void) __attribute__((requires_capability((Worker || Worker) && !GUI)));

int AlsoNotACapability;
void Func31(void) __attribute__((requires_capability(GUI && AlsoNotACapability))); // expected-warning {{'requires_capability' attribute requires arguments whose type is annotated with 'capability' attribute; type here is 'int'}}
