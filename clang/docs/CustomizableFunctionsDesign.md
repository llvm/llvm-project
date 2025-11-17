# Customizable Functions Design Document

**Author:** TBD
**Date:** 2025-11-17
**Status:** Draft

## Table of Contents

1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Proposed Syntax](#proposed-syntax)
4. [Semantics](#semantics)
5. [Implementation Plan](#implementation-plan)
6. [Testing Strategy](#testing-strategy)
7. [Open Questions](#open-questions)
8. [References](#references)

---

## Introduction

This document proposes adding a new language feature to Clang: **Customizable Functions** using the `custom` keyword. This feature would allow library authors to easily create Customization Point Objects (CPOs) using the tag_invoke pattern without manually writing extensive boilerplate code.

### Goals

- Reduce boilerplate for implementing the tag_invoke CPO pattern
- Improve library interface design and customization points
- Maintain compatibility with existing C++20/23 code
- Generate efficient, zero-overhead abstractions

### Non-Goals

- Replacing existing customization mechanisms (ADL, specialization)
- Changing the behavior of tag_invoke itself
- Adding runtime dispatch overhead

---

## Motivation

### Current State: Manual CPO Implementation

Library authors currently need to write significant boilerplate to create proper CPOs:

```cpp
namespace my_lib {
    namespace CPO_DETAIL {
        // Default implementation
        void do_thing(auto& t) { t.method(); }

        // Functor class
        struct do_thing_fn {
            // tag_invoke overload
            template <typename T>
            constexpr auto operator()(T& t) const
                requires requires { tag_invoke(*this, t); }
            {
                return tag_invoke(*this, t);
            }

            // Fallback overload
            template <typename T>
            constexpr auto operator()(T& t) const
                requires (!requires { tag_invoke(*this, t); })
            {
                return do_thing(t);
            }
        };
    }

    // Inline constexpr instance
    inline constexpr CPO_DETAIL::do_thing_fn do_thing{};
}
```

This pattern:
- Is verbose and error-prone
- Requires understanding of advanced C++ techniques
- Must be repeated for each customization point
- Is difficult to maintain and modify

### Proposed State: Declarative Syntax

With customizable functions, the same functionality becomes:

```cpp
namespace my_lib {
    custom void do_thing(auto& t) {
        t.method();
    }
}
```

The compiler automatically generates the CPO boilerplate, making the code:
- More concise and readable
- Less error-prone
- Easier to maintain
- Self-documenting

### Real-World Use Cases

1. **Generic Algorithm Libraries**: Customization points for ranges, algorithms
2. **Serialization/Deserialization**: Custom serializers for user types
3. **Logging/Debugging**: Customizable formatting and output
4. **Resource Management**: Custom allocators, deleters
5. **Async/Await Patterns**: Customizable awaitable operations

---

## Proposed Syntax

### Basic Syntax

```cpp
custom <return-type> <function-name>(<parameters>) <body>
```

### Examples

#### Simple Free Function
```cpp
namespace lib {
    custom void print(auto const& value) {
        std::cout << value;
    }
}
```

#### With Explicit Return Type
```cpp
custom int compute(int x, int y) {
    return x + y;
}
```

#### Multiple Parameters
```cpp
custom void transform(auto& container, auto&& func) {
    for (auto& elem : container) {
        func(elem);
    }
}
```

#### With Constraints
```cpp
custom void process(auto& t)
    requires std::copyable<decltype(t)>
{
    t.process();
}
```

#### Template Function
```cpp
template <typename T>
custom void serialize(T const& value, std::ostream& out) {
    out << value;
}
```

### Syntax Restrictions

- `custom` keyword must appear before the return type
- Cannot be used with:
  - Member functions (initially; may be relaxed later)
  - Constructors/destructors
  - Operators (except when implementing operator() for the generated functor)
  - Virtual functions
- Must have a function body (not just a declaration)
- Can be used in any namespace (including global)

---

## Semantics

### Transformation Overview

When the compiler encounters a `custom` function, it performs the following transformation:

1. **Create Detail Namespace** (optional, configurable)
   - Named `CPO_DETAIL` or `<function_name>_detail`
   - Contains the default implementation and functor class

2. **Generate Default Implementation**
   - Original function body becomes a hidden implementation function
   - Used as the fallback when tag_invoke is not available

3. **Generate Functor Class**
   - Named `<function_name>_fn`
   - Implements two operator() overloads:
     - Primary: calls tag_invoke (when available)
     - Fallback: calls default implementation

4. **Create CPO Instance**
   - Inline constexpr variable with original function name
   - Type is the functor class
   - This becomes the actual customization point

### Generated Code Structure

For a `custom` function:
```cpp
custom RetType func_name(Params...) { body }
```

The compiler generates:
```cpp
namespace CPO_DETAIL {
    // Default implementation (hidden)
    RetType func_name_impl(Params...) { body }

    // Functor class
    struct func_name_fn {
        // Primary overload: use tag_invoke if available
        template <typename... Args>
        constexpr auto operator()(Args&&... args) const
            noexcept(noexcept(tag_invoke(*this, std::forward<Args>(args)...)))
            requires requires { tag_invoke(*this, std::forward<Args>(args)...); }
        {
            return tag_invoke(*this, std::forward<Args>(args)...);
        }

        // Fallback overload: use default implementation
        template <typename... Args>
        constexpr auto operator()(Args&&... args) const
            noexcept(noexcept(func_name_impl(std::forward<Args>(args)...)))
            requires (!requires { tag_invoke(*this, std::forward<Args>(args)...); }
                      && requires { func_name_impl(std::forward<Args>(args)...); })
        {
            return func_name_impl(std::forward<Args>(args)...);
        }
    };
}

// The actual CPO
inline constexpr CPO_DETAIL::func_name_fn func_name{};
```

### Name Lookup and ADL

The generated CPO follows standard C++ name lookup rules:

1. **Unqualified calls** to the CPO trigger ADL
2. **tag_invoke** is found via ADL in the namespace of the arguments
3. Users customize by defining `tag_invoke` in their namespace:

```cpp
namespace user {
    struct MyType { };

    // Customization
    void tag_invoke(lib::do_thing_fn, MyType& t) {
        // Custom implementation
    }
}

// Usage
user::MyType obj;
lib::do_thing(obj);  // Finds user::tag_invoke via ADL
```

### Overload Resolution

The two operator() overloads in the functor are constrained to be mutually exclusive:
- Primary: `requires tag_invoke(*this, args...)`
- Fallback: `requires !tag_invoke(*this, args...) && default_impl(args...)`

This ensures:
- No ambiguity during overload resolution
- tag_invoke always preferred when available
- Fallback only selected when tag_invoke is not viable

### Template Instantiation

For templated custom functions:
```cpp
template <typename T>
custom void process(T& value) { ... }
```

The functor class itself is templated on the CPO's template parameters, and the operator() remains templated on the call-site arguments.

---

## Implementation Plan

### Phase 1: Core Infrastructure (Minimal Viable Product)

**Goal**: Get basic `custom` keyword parsing and simple code generation working.

#### 1.1 Add Keyword and Language Option

**Files**:
- `clang/include/clang/Basic/TokenKinds.def`
- `clang/include/clang/Basic/LangOptions.def`

**Tasks**:
- [ ] Add `custom` as a C++20 keyword with flag `KEYCUSTOMFN`
- [ ] Add language option `CustomizableFunctions` (default: disabled)
- [ ] Add `-fcustomizable-functions` / `-fno-customizable-functions` flags

**Testing**:
- [ ] Verify keyword is recognized when feature is enabled
- [ ] Verify keyword is not recognized when feature is disabled

#### 1.2 Extend Parser

**Files**:
- `clang/include/clang/Parse/Parser.h`
- `clang/lib/Parse/ParseDecl.cpp`
- `clang/include/clang/Sema/DeclSpec.h`

**Tasks**:
- [ ] Add `isCustomFunction` flag to `DeclSpec`
- [ ] Modify `ParseDeclarationSpecifiers` to recognize `custom` keyword
- [ ] Set the flag when `custom` is encountered
- [ ] Ensure proper error handling for invalid uses

**Testing**:
- [ ] Parse simple `custom void foo() {}`
- [ ] Reject `custom` on member functions
- [ ] Reject `custom` on declarations without definitions
- [ ] Proper error messages for invalid syntax

#### 1.3 Extend AST

**Files**:
- `clang/include/clang/AST/Decl.h`
- `clang/lib/AST/Decl.cpp`
- `clang/lib/AST/DeclPrinter.cpp`
- `clang/lib/AST/ASTDumper.cpp`

**Tasks**:
- [ ] Add `IsCustomFunction` bit to `FunctionDeclBitfields`
- [ ] Add `isCustomFunction()` / `setCustomFunction()` methods
- [ ] Update `DeclPrinter` to print `custom` keyword
- [ ] Update `ASTDumper` to show custom function flag

**Testing**:
- [ ] AST dump shows custom function annotation
- [ ] AST printer reproduces `custom` keyword

#### 1.4 Basic Semantic Analysis

**Files**:
- `clang/lib/Sema/SemaDecl.cpp`
- `clang/include/clang/Sema/Sema.h`

**Tasks**:
- [ ] Create `ActOnCustomFunctionDecl()` hook
- [ ] Validate custom function constraints:
  - Must have a body
  - Cannot be member function
  - Cannot be virtual
  - Cannot be main()
- [ ] Mark function as custom in AST

**Testing**:
- [ ] Semantic errors for invalid custom functions
- [ ] Accept valid custom function declarations

### Phase 2: Code Generation (Core Transformation)

**Goal**: Generate the CPO boilerplate during Sema.

#### 2.1 Create Sema Transformation Module

**Files** (new):
- `clang/include/clang/Sema/SemaCustomFunction.h`
- `clang/lib/Sema/SemaCustomFunction.cpp`

**Tasks**:
- [ ] Implement `class CustomFunctionTransformer`
- [ ] Add transformation entry point: `TransformCustomFunction(FunctionDecl*)`
- [ ] Generate detail namespace (optional, via flag)
- [ ] Generate default implementation function
- [ ] Generate functor class with operator() overloads
- [ ] Generate inline constexpr CPO instance
- [ ] Integrate into `ActOnFunctionDeclarator`

**Testing**:
- [ ] Generated AST contains all components
- [ ] Names are correct (e.g., `func_name_fn`)
- [ ] Proper linkage and storage for generated entities

#### 2.2 Functor Class Generation

**Tasks**:
- [ ] Create `CXXRecordDecl` for functor class
- [ ] Add `operator()` with tag_invoke call
- [ ] Add `operator()` with fallback call
- [ ] Generate proper constraints (requires clauses)
- [ ] Handle noexcept specifications
- [ ] Handle return type deduction

**Testing**:
- [ ] Functor class has correct structure
- [ ] Overload resolution works correctly
- [ ] Constraints are mutually exclusive

#### 2.3 CPO Instance Generation

**Tasks**:
- [ ] Create `VarDecl` for CPO instance
- [ ] Set proper storage class (inline constexpr)
- [ ] Initialize with functor class instance
- [ ] Handle name shadowing of original function

**Testing**:
- [ ] CPO instance has correct type
- [ ] CPO instance is constexpr
- [ ] Original function name resolves to CPO instance

### Phase 3: Template Support

**Goal**: Handle template custom functions correctly.

#### 3.1 Template Function Transformation

**Tasks**:
- [ ] Handle `template <...> custom void foo(...)`
- [ ] Preserve template parameters on generated entities
- [ ] Handle template instantiation correctly
- [ ] Support template constraints

**Testing**:
- [ ] Template custom functions instantiate correctly
- [ ] Template argument deduction works
- [ ] SFINAE works with custom functions
- [ ] Constraints propagate correctly

#### 3.2 Template Parameter Forwarding

**Tasks**:
- [ ] Forward template parameters to functor class
- [ ] Ensure proper argument forwarding in operator()
- [ ] Handle perfect forwarding scenarios

**Testing**:
- [ ] Forwarding references work correctly
- [ ] Move semantics preserved
- [ ] No extra copies/moves

### Phase 4: Advanced Features

#### 4.1 Customization Options

Add attributes or keywords to control generation:

```cpp
[[custom::no_detail_namespace]]
custom void foo() { }

[[custom::detail_namespace("my_detail")]]
custom void bar() { }
```

**Tasks**:
- [ ] Parse customization attributes
- [ ] Support custom detail namespace names
- [ ] Support option to skip detail namespace
- [ ] Support custom functor class naming

#### 4.2 Diagnostics and Error Messages

**Tasks**:
- [ ] Improve error messages for invalid customizations
- [ ] Add notes pointing to original custom function
- [ ] Handle recursive tag_invoke calls
- [ ] Detect and warn about common mistakes

**Testing**:
- [ ] Error messages are clear and helpful
- [ ] Source locations point to correct code
- [ ] Notes provide useful context

#### 4.3 Integration with TInCuP Library

**Tasks**:
- [ ] Option to generate TInCuP-compatible code
- [ ] Add `tincup::cpo_base` inheritance (optional)
- [ ] Generate concept traits (e.g., `foo_invocable_c`)
- [ ] Generate type traits (e.g., `foo_return_t`)

### Phase 5: Tooling and Serialization

#### 5.1 Modules and PCH Support

**Files**:
- `clang/include/clang/Serialization/ASTWriter.h`
- `clang/include/clang/Serialization/ASTReader.h`

**Tasks**:
- [ ] Serialize custom function flag
- [ ] Serialize generated entities
- [ ] Handle cross-module references

**Testing**:
- [ ] Custom functions work in modules
- [ ] PCH files handle custom functions
- [ ] Module imports work correctly

#### 5.2 Code Completion and IDE Support

**Tasks**:
- [ ] Code completion for custom functions
- [ ] Signature help shows generated CPO
- [ ] Go-to-definition works for both original and generated code

#### 5.3 Debug Info Generation

**Tasks**:
- [ ] Generate proper debug info for all entities
- [ ] Debugger can step into both tag_invoke and fallback
- [ ] Variable inspection works correctly

---

## Testing Strategy

### Unit Tests

#### Lexer/Parser Tests
**Location**: `clang/test/Parser/cxx-customizable-functions.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s

// Basic parsing
custom void foo() { }  // OK

// Reject invalid uses
custom int x = 5;  // expected-error {{expected function}}
class C {
    custom void bar();  // expected-error {{custom functions cannot be member functions}}
};
```

#### Semantic Analysis Tests
**Location**: `clang/test/SemaCXX/customizable-functions-sema.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s

namespace test1 {
    custom void func(int x) { }  // OK
}

namespace test2 {
    custom void forward_decl();  // expected-error {{custom functions must have a body}}
}

namespace test3 {
    struct S {
        custom void member();  // expected-error {{custom functions cannot be member functions}}
    };
}
```

#### AST Tests
**Location**: `clang/test/AST/customizable-functions-ast.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -ast-dump %s | FileCheck %s

custom void test() { }

// CHECK: FunctionDecl {{.*}} test 'void ()'
// CHECK-NEXT: CustomFunction
// CHECK: NamespaceDecl {{.*}} CPO_DETAIL
// CHECK: CXXRecordDecl {{.*}} test_fn
// CHECK: VarDecl {{.*}} test
```

### Integration Tests

#### Code Generation Tests
**Location**: `clang/test/CodeGenCXX/customizable-functions-codegen.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -emit-llvm %s -o - | FileCheck %s

custom void simple() { }

// CHECK: @_ZN10CPO_DETAIL9simple_fnE =
// CHECK: define {{.*}} @_ZN10CPO_DETAIL11simple_implEv()
```

#### Tag Invoke Integration Tests
**Location**: `clang/test/SemaCXX/customizable-functions-tag-invoke.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s

namespace lib {
    custom void process(auto& x) {
        x.default_process();
    }
}

namespace user {
    struct MyType {
        void default_process() { }
        void custom_process() { }
    };

    // Custom implementation
    void tag_invoke(lib::process_fn, MyType& m) {
        m.custom_process();
    }
}

void test() {
    user::MyType obj;
    lib::process(obj);  // Should call user::tag_invoke
}

// Should compile without errors
```

#### Template Tests
**Location**: `clang/test/SemaCXX/customizable-functions-templates.cpp`

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s

template <typename T>
custom void serialize(T const& value) {
    // default implementation
}

namespace user {
    struct MyType { };

    template <typename T>
    void tag_invoke(serialize_fn, MyType const& m) {
        // custom implementation
    }
}

void test() {
    user::MyType obj;
    serialize(obj);  // Should work
}
```

### End-to-End Tests

#### Complete Example Test
**Location**: `clang/test/SemaCXX/customizable-functions-e2e.cpp`

Full example with:
- Multiple custom functions
- User customizations via tag_invoke
- Overload resolution
- Template instantiation
- Constraint checking

### Performance Tests

#### Compilation Performance
- Measure compile time impact of custom functions
- Compare with manual CPO implementation
- Benchmark template instantiation overhead

#### Runtime Performance
- Verify zero-overhead abstraction
- Compare generated code with manual implementation
- Benchmark with optimization levels -O0, -O2, -O3

### Regression Tests

Add tests for:
- Previously reported bugs
- Edge cases discovered during development
- Interaction with other language features

---

## Open Questions

### 1. Detail Namespace Strategy

**Question**: Should we always generate a detail namespace, or make it optional?

**Options**:
- **A**: Always generate `CPO_DETAIL` namespace (consistent, but verbose)
- **B**: Optional via attribute: `[[custom::detail_namespace]]`
- **C**: Generate in same namespace without detail (simpler, but pollutes namespace)

**Recommendation**: Option B - optional with default enabled

### 2. Functor Class Naming

**Question**: What naming convention for generated functor class?

**Options**:
- **A**: `<name>_fn` (matches TInCuP)
- **B**: `<name>_functor`
- **C**: `__<name>_cpo_impl` (clearly internal)

**Recommendation**: Option A - matches existing conventions

### 3. Template Parameter Handling

**Question**: How to handle template custom functions?

**Options**:
- **A**: Functor class is templated, operator() is also templated
- **B**: Functor class is not templated, operator() is fully templated
- **C**: Generate a function template instead of functor class

**Recommendation**: Option A - most flexible

### 4. Member Function Support

**Question**: Should we support custom member functions?

**Options**:
- **A**: Never support (simplest)
- **B**: Support in future version (phase 2+)
- **C**: Support from the start

**Recommendation**: Option B - defer to future

**Rationale**: Member CPOs have different semantics and use cases. Start with free functions, add members later if needed.

### 5. Interaction with Concepts

**Question**: How should `requires` clauses on custom functions behave?

```cpp
custom void foo(auto x) requires std::integral<decltype(x)> { }
```

**Options**:
- **A**: Constraint applies to both tag_invoke and fallback
- **B**: Constraint only applies to fallback
- **C**: Generate separate constraints for each

**Recommendation**: Option A - constraint is part of the interface

### 6. Noexcept Specifications

**Question**: How to handle noexcept on custom functions?

```cpp
custom void foo() noexcept { }
```

**Options**:
- **A**: Conditional noexcept in generated operator()
- **B**: Unconditional noexcept in operator()
- **C**: Ignore and always deduce

**Recommendation**: Option A - preserve user intent

### 7. Default Arguments

**Question**: Should custom functions support default arguments?

```cpp
custom void foo(int x = 0) { }
```

**Options**:
- **A**: Support (forward to operator())
- **B**: Reject (error)
- **C**: Support only in fallback

**Recommendation**: Option A - forward to both paths

### 8. Customization Attributes

**Question**: What attributes should we support for customization?

**Proposals**:
- `[[custom::no_detail_namespace]]`
- `[[custom::detail_namespace("name")]]`
- `[[custom::functor_name("name")]]`
- `[[custom::tincup_compatible]]`
- `[[custom::inline_implementation]]`

**Recommendation**: Start with namespace control, add others as needed

---

## References

### Academic Papers and Proposals

1. **P1895R0**: tag_invoke: A general pattern for supporting customizable functions
   https://wg21.link/p1895r0

2. **P2279R0**: We need a language mechanism for customization points
   https://wg21.link/p2279r0

3. **P0443R14**: A Unified Executors Proposal for C++
   (Uses tag_invoke extensively)

### Related Projects

1. **TInCuP**: Tag Invoke Customization Points Library
   https://github.com/sandialabs/TInCuP

2. **niebloids and customization points**:
   https://brevzin.github.io/c++/2020/12/19/cpo-niebloid/

### LLVM/Clang Resources

1. **Coroutines Implementation**: `clang/lib/Sema/SemaCoroutine.cpp`
2. **Modules Implementation**: `clang/lib/Sema/SemaModule.cpp`
3. **Concepts Implementation**: `clang/lib/Sema/SemaConcept.cpp`

### Similar Features in Other Languages

1. **Rust Traits**: Customizable behavior via trait implementations
2. **Haskell Type Classes**: Similar customization mechanism
3. **Swift Extensions**: Protocol-based customization

---

## Appendix A: Example Transformations

### Example 1: Simple Function

**Input**:
```cpp
namespace lib {
    custom void log(auto const& msg) {
        std::cout << msg << std::endl;
    }
}
```

**Generated**:
```cpp
namespace lib {
    namespace CPO_DETAIL {
        void log_impl(auto const& msg) {
            std::cout << msg << std::endl;
        }

        struct log_fn {
            template <typename T>
            constexpr auto operator()(T const& msg) const
                noexcept(noexcept(tag_invoke(*this, msg)))
                requires requires { tag_invoke(*this, msg); }
            {
                return tag_invoke(*this, msg);
            }

            template <typename T>
            constexpr auto operator()(T const& msg) const
                noexcept(noexcept(log_impl(msg)))
                requires (!requires { tag_invoke(*this, msg); })
            {
                return log_impl(msg);
            }
        };
    }

    inline constexpr CPO_DETAIL::log_fn log{};
}
```

### Example 2: Template Function

**Input**:
```cpp
template <typename Stream>
custom void write(Stream& stream, auto const& data) {
    stream << data;
}
```

**Generated**:
```cpp
template <typename Stream>
struct write_fn {
    template <typename S, typename T>
    constexpr auto operator()(S& stream, T const& data) const
        requires requires { tag_invoke(*this, stream, data); }
    {
        return tag_invoke(*this, stream, data);
    }

    template <typename S, typename T>
    constexpr auto operator()(S& stream, T const& data) const
        requires (!requires { tag_invoke(*this, stream, data); })
    {
        return write_impl(stream, data);
    }
};

template <typename Stream>
inline constexpr write_fn<Stream> write{};

// Implementation
template <typename Stream>
void write_impl(Stream& stream, auto const& data) {
    stream << data;
}
```

### Example 3: With Constraints

**Input**:
```cpp
custom void process(auto& container)
    requires requires { container.begin(); container.end(); }
{
    for (auto& elem : container) {
        elem.update();
    }
}
```

**Generated** (constraints preserved in both paths):
```cpp
namespace CPO_DETAIL {
    void process_impl(auto& container)
        requires requires { container.begin(); container.end(); }
    {
        for (auto& elem : container) {
            elem.update();
        }
    }

    struct process_fn {
        template <typename T>
        constexpr auto operator()(T& container) const
            requires requires { container.begin(); container.end(); }
                  && requires { tag_invoke(*this, container); }
        {
            return tag_invoke(*this, container);
        }

        template <typename T>
        constexpr auto operator()(T& container) const
            requires requires { container.begin(); container.end(); }
                  && (!requires { tag_invoke(*this, container); })
        {
            return process_impl(container);
        }
    };
}

inline constexpr CPO_DETAIL::process_fn process{};
```

---

## Appendix B: Timeline and Milestones

### Milestone 1: Proof of Concept (2-3 weeks)
- [ ] Keyword parsing
- [ ] Basic AST representation
- [ ] Simple transformation (single free function)
- [ ] Minimal test suite

**Success Criteria**: Can compile and run a simple custom function example

### Milestone 2: Core Functionality (4-6 weeks)
- [ ] Complete transformation logic
- [ ] Template support
- [ ] Comprehensive test suite
- [ ] Basic diagnostics

**Success Criteria**: Can handle most common use cases correctly

### Milestone 3: Production Ready (8-12 weeks)
- [ ] Advanced features
- [ ] Optimized codegen
- [ ] Full diagnostic support
- [ ] Documentation
- [ ] Integration tests

**Success Criteria**: Ready for experimental use in real codebases

### Milestone 4: Stable Release (12-16 weeks)
- [ ] Modules/PCH support
- [ ] Tooling integration
- [ ] Performance optimization
- [ ] Community feedback integration

**Success Criteria**: Ready for upstream submission to LLVM

---

## Appendix C: Potential Issues and Mitigations

### Issue 1: Name Shadowing

**Problem**: The CPO instance shadows the original function name.

**Mitigation**: This is intentional - users call the CPO, not the original function.

### Issue 2: Overload Set Conflicts

**Problem**: Multiple custom functions with same name but different signatures.

**Mitigation**: Generate a single functor class with multiple operator() overloads, similar to std::ranges algorithms.

### Issue 3: Template Instantiation Bloat

**Problem**: Each custom template function generates multiple templates.

**Mitigation**:
- Inline small functions
- Use `if constexpr` where possible
- Rely on linker to deduplicate

### Issue 4: Error Message Quality

**Problem**: Errors in generated code may confuse users.

**Mitigation**:
- Always point diagnostics to original source
- Add "note: in generated code for custom function 'X'"
- Suppress internal implementation details in errors

### Issue 5: Compilation Time

**Problem**: More templates may slow compilation.

**Mitigation**:
- Measure and profile
- Optimize common patterns
- Consider precompiled headers for heavy users

---

## Revision History

- **2025-11-17**: Initial draft
