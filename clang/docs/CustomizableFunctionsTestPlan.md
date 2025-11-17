# Customizable Functions Test Plan

This document provides a detailed testing strategy for the Customizable Functions feature.

## Test Categories

### 1. Parser Tests

#### 1.1 Keyword Recognition
**File**: `clang/test/Parser/cxx-customizable-functions-keyword.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s

// Test basic keyword recognition
custom void test1() { }  // OK

// Test without flag enabled
// RUN: %clang_cc1 -std=c++20 -fno-customizable-functions -fsyntax-only -verify=disabled %s
// disabled-error@-4 {{unknown type name 'custom'}}

// Test with older C++ standard
// RUN: %clang_cc1 -std=c++17 -fcustomizable-functions -fsyntax-only -verify=cpp17 %s
// cpp17-warning@-8 {{'custom' is a C++20 extension}}
```

#### 1.2 Syntax Validation
**File**: `clang/test/Parser/cxx-customizable-functions-syntax.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s

// Valid uses
custom void f1() { }
custom int f2() { return 0; }
custom auto f3() { return 42; }
custom void f4(int x) { }
custom void f5(auto x) { }

namespace ns {
    custom void f6() { }
}

// Invalid: no body
custom void f7();  // expected-error {{custom functions must have a definition}}

// Invalid: custom on non-function
custom int x = 5;  // expected-error {{expected function}}
custom;  // expected-error {{expected declaration}}

// Invalid: multiple storage classes
static custom void f8() { }  // expected-error {{'custom' cannot be combined with 'static'}}
extern custom void f9() { }  // expected-error {{'custom' cannot be combined with 'extern'}}
```

#### 1.3 Declaration Specifier Ordering
**File**: `clang/test/Parser/cxx-customizable-functions-declspec-order.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s

// Valid: custom before return type
custom void f1() { }
custom int f2() { return 0; }
custom constexpr int f3() { return 0; }

// Valid: inline with custom
inline custom void f4() { }
custom inline void f5() { }

// Invalid: custom after return type
void custom f6() { }  // expected-error {{expected identifier}}
int custom f7() { return 0; }  // expected-error {{expected identifier}}
```

### 2. Semantic Analysis Tests

#### 2.1 Context Restrictions
**File**: `clang/test/SemaCXX/customizable-functions-context.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s

// OK: namespace scope
namespace ns {
    custom void f1() { }
}

// OK: global scope
custom void f2() { }

// ERROR: class member
struct S1 {
    custom void f3() { }  // expected-error {{custom functions cannot be member functions}}
    custom static void f4() { }  // expected-error {{custom functions cannot be member functions}}
};

// ERROR: local scope
void test() {
    custom void f5() { }  // expected-error {{custom functions must be declared at namespace scope}}
}

// ERROR: special members
struct S2 {
    custom S2() { }  // expected-error {{constructors cannot be custom}}
    custom ~S2() { }  // expected-error {{destructors cannot be custom}}
    custom operator int() { return 0; }  // expected-error {{conversion functions cannot be custom}}
};

// ERROR: virtual
struct S3 {
    custom virtual void f6() { }  // expected-error {{custom functions cannot be virtual}}
};

// ERROR: main function
custom int main() { return 0; }  // expected-error {{'main' cannot be a custom function}}
```

#### 2.2 Template Validation
**File**: `clang/test/SemaCXX/customizable-functions-templates.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s

// OK: function template
template <typename T>
custom void f1(T x) { }

// OK: multiple template parameters
template <typename T, typename U>
custom void f2(T x, U y) { }

// OK: template with constraints
template <typename T>
    requires std::integral<T>
custom void f3(T x) { }

// OK: abbreviated function template
custom void f4(auto x) { }

// ERROR: member function template
struct S {
    template <typename T>
    custom void f5(T x) { }  // expected-error {{custom functions cannot be member functions}}
};

// OK: template specialization (becomes specialized custom function)
template <>
custom void f1<int>(int x) { }

// Test instantiation
void test() {
    f1(42);      // OK
    f1(3.14);    // OK
    f2(1, 2);    // OK
    f3(5);       // OK
    f4("hello"); // OK
}
```

#### 2.3 Type and Return Type Deduction
**File**: `clang/test/SemaCXX/customizable-functions-deduction.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s

// OK: explicit return type
custom int f1() { return 42; }

// OK: auto return type
custom auto f2() { return 42; }

// OK: decltype(auto)
custom decltype(auto) f3() { return 42; }

// OK: trailing return type
custom auto f4() -> int { return 42; }

// OK: template return type deduction
template <typename T>
custom auto f5(T x) { return x; }

// ERROR: inconsistent return
custom auto f6(bool b) {
    if (b)
        return 42;     // int
    else
        return 3.14;   // expected-error {{return type deduction conflicts}}
}

// Test return type propagation
void test() {
    static_assert(std::is_same_v<decltype(f1()), int>);
    static_assert(std::is_same_v<decltype(f2()), int>);
    static_assert(std::is_same_v<decltype(f5(42)), int>);
}
```

### 3. Code Generation Tests

#### 3.1 Basic CPO Generation
**File**: `clang/test/CodeGenCXX/customizable-functions-basic-codegen.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s

custom void simple() { }

// CHECK: @[[INSTANCE:_ZN10CPO_DETAIL9simple_fnE]] = linkonce_odr global
// CHECK: define {{.*}} @[[IMPL:_ZN10CPO_DETAIL11simple_implEv]]()

// CHECK: define {{.*}} @[[FALLBACK:.*]]()
// CHECK: call {{.*}} @[[IMPL]]()

// Verify functor class structure
// CHECK: %[[FUNCTOR:.*]] = type { i8 }
```

#### 3.2 Tag Invoke Call
**File**: `clang/test/CodeGenCXX/customizable-functions-tag-invoke.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s

namespace lib {
    custom void process(auto& x) {
        x.default_impl();
    }
}

namespace user {
    struct MyType {
        void default_impl();
        void custom_impl();
    };

    // User customization
    void tag_invoke(lib::process_fn, MyType& m) {
        m.custom_impl();
    }
}

void test() {
    user::MyType obj;
    lib::process(obj);
}

// CHECK: define {{.*}} @_ZN4user10tag_invokeEN3lib10process_fnERNS_6MyTypeE
// CHECK: call {{.*}} @_ZN4user6MyType11custom_implEv
```

#### 3.3 Template Instantiation
**File**: `clang/test/CodeGenCXX/customizable-functions-template-codegen.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s

template <typename T>
custom void print(T const& value) {
    // default implementation
}

void test() {
    print(42);
    print(3.14);
    print("hello");
}

// CHECK: define {{.*}} @{{.*}}print{{.*}}i{{.*}}
// CHECK: define {{.*}} @{{.*}}print{{.*}}d{{.*}}
// CHECK: define {{.*}} @{{.*}}print{{.*}}PKc{{.*}}
```

### 4. Integration Tests

#### 4.1 Complete Tag Invoke Workflow
**File**: `clang/test/SemaCXX/customizable-functions-integration.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s
// expected-no-diagnostics

#include <iostream>
#include <string>

// Library defines customization point
namespace lib {
    custom void serialize(auto const& value, std::ostream& out) {
        out << value;  // default: use operator<<
    }
}

// User type with customization
namespace user {
    struct Person {
        std::string name;
        int age;
    };

    // User provides custom serialization
    void tag_invoke(lib::serialize_fn, Person const& p, std::ostream& out) {
        out << "Person{name=" << p.name << ", age=" << p.age << "}";
    }
}

// Test that it all works
void test() {
    // Built-in type uses default implementation
    lib::serialize(42, std::cout);

    // User type uses custom implementation
    user::Person p{"Alice", 30};
    lib::serialize(p, std::cout);

    // String uses default implementation
    lib::serialize(std::string("hello"), std::cout);
}
```

#### 4.2 ADL and Overload Resolution
**File**: `clang/test/SemaCXX/customizable-functions-adl.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s
// expected-no-diagnostics

namespace lib {
    custom void process(auto& x) {
        x.process();
    }
}

namespace ns1 {
    struct Type1 {
        void process() { }
    };

    void tag_invoke(lib::process_fn, Type1& t) {
        // Custom implementation for ns1::Type1
    }
}

namespace ns2 {
    struct Type2 {
        void process() { }
    };

    void tag_invoke(lib::process_fn, Type2& t) {
        // Custom implementation for ns2::Type2
    }
}

void test() {
    ns1::Type1 t1;
    ns2::Type2 t2;

    // ADL finds the right tag_invoke in each namespace
    lib::process(t1);  // calls ns1::tag_invoke
    lib::process(t2);  // calls ns2::tag_invoke
}
```

#### 4.3 Constraints and SFINAE
**File**: `clang/test/SemaCXX/customizable-functions-constraints.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s

#include <concepts>

// Constrained custom function
custom void process_integral(auto x)
    requires std::integral<decltype(x)>
{
    // process integral type
}

namespace user {
    struct NotIntegral { };

    // This should not be selected even with tag_invoke
    void tag_invoke(process_integral_fn, NotIntegral x) { }
}

void test() {
    process_integral(42);           // OK
    process_integral('a');          // OK
    process_integral(true);         // OK

    user::NotIntegral ni;
    process_integral(ni);           // expected-error {{no matching function}}
    process_integral(3.14);         // expected-error {{no matching function}}
}
```

### 5. AST Tests

#### 5.1 AST Structure
**File**: `clang/test/AST/customizable-functions-ast-dump.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -ast-dump %s | FileCheck %s

custom void test() { }

// CHECK: FunctionDecl {{.*}} test 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK: NamespaceDecl {{.*}} CPO_DETAIL
// CHECK-NEXT: FunctionDecl {{.*}} referenced test_impl 'void ()'
// CHECK-NEXT: CXXRecordDecl {{.*}} test_fn
// CHECK-NEXT: CXXMethodDecl {{.*}} operator() 'auto (auto) const'
// CHECK-NEXT: CXXMethodDecl {{.*}} operator() 'auto (auto) const'
// CHECK: VarDecl {{.*}} test 'const test_fn':'const CPO_DETAIL::test_fn' inline constexpr
```

#### 5.2 AST Printing
**File**: `clang/test/AST/customizable-functions-ast-print.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -ast-print %s | FileCheck %s

custom void test() { }

// CHECK: custom void test() {
// CHECK: }

// CHECK: namespace CPO_DETAIL {
// CHECK: void test_impl()
// CHECK: struct test_fn {
// CHECK: }
// CHECK: inline constexpr test_fn test;
```

### 6. Diagnostic Tests

#### 6.1 Error Messages
**File**: `clang/test/SemaCXX/customizable-functions-errors.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s

// Test clear error messages

struct S {
    custom void member() { }
    // expected-error@-1 {{custom functions cannot be member functions}}
    // expected-note@-2 {{remove 'custom' to declare a regular member function}}
};

custom void no_body();
// expected-error@-1 {{custom functions must have a definition}}
// expected-note@-2 {{add a function body}}

static custom void with_static() { }
// expected-error@-1 {{'custom' cannot be combined with 'static'}}

custom virtual void with_virtual() { }
// expected-error@-1 {{custom functions cannot be virtual}}
```

#### 6.2 Warning Messages
**File**: `clang/test/SemaCXX/customizable-functions-warnings.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s

// Test warnings

custom void unused() { }
// expected-warning@-1 {{custom function 'unused' is never used}}

custom void recursive(auto& x) {
    recursive(x);  // expected-warning {{recursive call to custom function may cause infinite loop}}
}
```

### 7. Edge Cases and Corner Cases

#### 7.1 Name Conflicts
**File**: `clang/test/SemaCXX/customizable-functions-name-conflicts.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s

namespace test1 {
    custom void func() { }
    void func_fn() { }  // expected-error {{redefinition of 'func_fn'}}
                        // expected-note@-3 {{previous definition generated for custom function 'func'}}
}

namespace test2 {
    custom void func() { }
    namespace CPO_DETAIL {  // expected-error {{redefinition of 'CPO_DETAIL'}}
                            // expected-note@-2 {{previous definition generated for custom function 'func'}}
    }
}
```

#### 7.2 Nested Namespaces
**File**: `clang/test/SemaCXX/customizable-functions-nested-namespaces.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s
// expected-no-diagnostics

namespace outer {
    namespace inner {
        custom void func() { }
    }
}

void test() {
    outer::inner::func();  // OK
}
```

#### 7.3 Forward References
**File**: `clang/test/SemaCXX/customizable-functions-forward-refs.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s

// Can we reference the CPO before it's fully defined?
namespace test {
    void use_it();  // declares function that uses CPO

    custom void cpo() { }

    void use_it() {
        cpo();  // OK
    }
}
```

### 8. Performance Tests

#### 8.1 Zero Overhead
**File**: `clang/test/CodeGenCXX/customizable-functions-zero-overhead.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -triple x86_64-linux-gnu -emit-llvm -O2 %s -o - | FileCheck %s

custom void simple(int& x) {
    x++;
}

void test(int& x) {
    simple(x);
}

// With -O2, should inline to just x++
// CHECK-LABEL: @_Z4testRi
// CHECK: add nsw i32 {{.*}}, 1
// CHECK-NEXT: store
// CHECK-NEXT: ret
```

#### 8.2 Compile Time
**File**: `clang/test/CodeGenCXX/customizable-functions-compile-time.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -ftime-report %s -o /dev/null 2>&1 | FileCheck %s

// Many custom functions
custom void f1() { }
custom void f2() { }
custom void f3() { }
// ... many more ...
custom void f100() { }

// CHECK: Custom Function Transformation
// CHECK: Total Execution Time: {{[0-9.]+}}
```

### 9. Module and PCH Tests

#### 9.1 Modules
**File**: `clang/test/Modules/customizable-functions.cppm`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -emit-module-interface %s -o %t.pcm
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fmodule-file=%t.pcm -verify %s

export module test;

export custom void exported_cpo() { }

// Test that importing works
// RUN: echo 'import test; void use() { exported_cpo(); }' | %clang_cc1 -std=c++20 -fcustomizable-functions -fmodule-file=%t.pcm -x c++ - -fsyntax-only -verify
```

#### 9.2 Precompiled Headers
**File**: `clang/test/PCH/customizable-functions.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -emit-pch %s -o %t.pch
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -include-pch %t.pch -verify %s

#ifndef HEADER
#define HEADER

custom void in_header() { }

#else

void test() {
    in_header();  // OK
}

#endif
```

### 10. Standard Library Integration Tests

#### 10.1 Ranges Integration
**File**: `clang/test/SemaCXX/customizable-functions-ranges.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s
// expected-no-diagnostics

#include <ranges>
#include <vector>

namespace lib {
    custom void for_each(auto&& range, auto&& func)
        requires std::ranges::range<decltype(range)>
    {
        for (auto&& elem : range) {
            func(elem);
        }
    }
}

void test() {
    std::vector<int> v = {1, 2, 3};
    lib::for_each(v, [](int x) { });  // OK
}
```

### 11. Regression Tests

Tests for specific bugs found during development.

**File**: `clang/test/SemaCXX/customizable-functions-regression.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s
// expected-no-diagnostics

// Add regression tests as bugs are found and fixed
```

---

## Test Execution Strategy

### Continuous Testing

1. **Pre-commit**: Run parser and semantic tests
2. **Post-commit**: Run full test suite including codegen
3. **Nightly**: Run performance benchmarks
4. **Release**: Run extensive integration tests

### Test Coverage Goals

- **Parser**: 100% coverage of syntax cases
- **Sema**: 100% coverage of semantic rules
- **CodeGen**: 90%+ coverage of generation patterns
- **Integration**: Cover all major use cases

### Performance Benchmarks

Create benchmarks comparing:
1. Manual CPO implementation vs. generated
2. Compilation time impact
3. Runtime performance (should be identical)
4. Binary size impact

### Test Infrastructure

**Tools**:
- `FileCheck` for verifying compiler output
- `not` for verifying errors
- `diff` for comparing generated code
- Custom lit tests for end-to-end scenarios

**Test Organization**:
```
clang/test/
├── Parser/
│   ├── cxx-customizable-functions-*.cpp
├── SemaCXX/
│   ├── customizable-functions-*.cpp
├── CodeGenCXX/
│   ├── customizable-functions-*.cpp
├── AST/
│   ├── customizable-functions-*.cpp
├── Modules/
│   ├── customizable-functions.cppm
└── PCH/
    ├── customizable-functions.cpp
```

---

## Acceptance Criteria

Before merging, the feature must:

1. **Pass all tests**: 100% test pass rate
2. **No regressions**: All existing tests still pass
3. **Performance neutral**: No measurable slowdown in non-custom code
4. **Documentation**: Complete user documentation
5. **Code review**: Approved by LLVM community
6. **Examples**: Working examples for common use cases

---

## Future Test Considerations

Tests to add in future phases:

1. **Member function support** (if added)
2. **Attributes and customization options**
3. **TInCuP library integration**
4. **Cross-platform testing** (Windows, macOS, Linux)
5. **Debugger integration** (LLDB, GDB)
6. **Static analyzer support**
7. **Code completion and IDE integration**

---

## Test Metrics

Track these metrics during development:

- **Test count**: Number of test cases
- **Code coverage**: Percentage of new code covered
- **Bug count**: Bugs found via testing
- **Regression count**: Regressions caught
- **Performance**: Compile time impact

Target: 90%+ code coverage, 0 known regressions before release.
