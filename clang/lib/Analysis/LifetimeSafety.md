Excellent! This is a very strong and logical structure for the white paper. It follows a clear narrative, starting from the high-level problem and progressively diving into the specifics of your solution. The sections on why a traditional borrow checker doesn't fit C++ and the open questions are particularly good, as they show a deep engagement with the problem space.

Here is a draft of the white paper following your new skeleton, with the details filled in based on my analysis of your implementation and the provided reference documents. I've also incorporated some of my own suggestions to enhance the flow and clarity.

***

<Disclaimer: Public document. This work is licensed under the Apache License v2.0 with LLVM Exceptions. See [https://llvm.org/LICENSE.txt](https://llvm.org/LICENSE.txt) for license information.>

# Lifetime Safety: An Intuitive Approach for Temporal Safety in C++
**Author:**
[Utkarsh Saxena](mailto:usx@google.com)

**Purpose:** This document serves as a live RFC for a new lifetime safety analysis in C++, with the ultimate goal of publication as a white paper.

## Intended Audience

This document is intended for C++ compiler developers (especially those working on Clang), developers of other systems languages with advanced memory safety models (like Rust and Carbon), and all C++ users interested in writing safer code.

## Goal

*   To describe a new lifetime model for C++ that aims to maximize the compile-time detection of temporal memory safety issues.
*   To explore a path toward incremental safety in C++, offering a spectrum of checks that can be adopted without requiring a full plunge into a restrictive ownership model.

**Out of Scope**

*   **Rigorous Temporal Memory Safety:** This analysis aims to detect a large class of common errors, but it does not formally prove the absence of all temporal safety bugs.
*   **Runtime Solutions:** This paper focuses exclusively on static, compile-time analysis and does not cover runtime solutions like MTE or AddressSanitizer.

# Paper: C++ Lifetimes Safety Analysis

**Subtitle: A Flow-Sensitive, Alias-based Approach to Preventing Dangling Pointers**

## Abstract

This paper introduces a new intra-procedural, flow-sensitive lifetime analysis for C++ implemented in Clang. The analysis is designed to detect a significant class of temporal memory safety violations, such as use-after-free and use-after-return, at compile time. It is based on a model of "Loans" and "Origins," inspired by the Polonius borrow checker in Rust, but adapted for the semantics and flexibility of C++.

The analysis works by translating the Clang CFG into a series of lifetime-relevant "Facts." These facts are then processed by dataflow analyses to precisely determine the validity of pointers and references at each program point. This fact-based approach, combined with a configurable strictness model, allows for both high-confidence error reporting and the detection of more subtle, potential bugs, without requiring extensive new annotations. The ultimate goal is to provide a powerful, low-overhead tool that makes C++ safer by default.

## The Anatomy of a Temporal Safety Error

At its core, a temporal safety error is a bug where an operation is performed on an object at a time when it is no longer valid to do so ([source](http://docs.google.com/document/d/19vbfAiV1yQu3xSMRWjyPUdzyB_LDdVUcKat_HWI1l3g?content_ref=at+its+core+a+temporal+safety+error+is+a+bug+where+an+operation+is+performed+on+an+object+at+a+time+when+it+is+no+longer+valid+to+do+so)). These bugs are notoriously difficult to debug because they often manifest as unpredictable crashes or silent data corruption far from the root cause. However, we can argue that this wide and varied class of errors—from use-after-free to iterator invalidation—all stem from a single, canonical pattern.

**Conjecture: Any temporal safety issue is a form of Use-After-Free.**

All sub-categories of temporal safety issues, such as returning a reference to a stack variable (`return-stack-addr`), using a variable after its scope has ended (`use-after-scope`), using heap memory after it has been deleted (`heap-use-after-free`), or using an iterator after its container has been modified (`use-after-invalidation`), can be described by a single sequence of events.

In C++, an *object* is a region of storage, and pointers and references are the mechanisms we use to refer to them. A use-after-free occurs when we access an object after its lifetime has ended. But how can an object be accessed after it has been destroyed? This is only possible through an **alias**—a pointer or reference—that was created while the object was alive and that survived the object's destruction.

This insight allows us to define a canonical use-after-free with four distinct events that happen in a specific order:

1.  **`t0`: Creation.** An object `M` is created in some region of storage (on the stack, on the heap, etc.).
2.  **`t1`: Alias Creation.** An alias `P` (a pointer or reference) is created that refers to the object `M`.
3.  **`t2`: End of Lifetime.** The lifetime of object `M` ends (e.g., it is deallocated, or it goes out of scope).
4.  **`t3`: Use of Alias.** The alias `P`, which now dangles, is used to access the memory where `M` once resided.

Let's examine this with a simple piece of C++ code:

```cpp
void use_after_scope_example() {
  int* p;
  {
    int s = 10;  // t0: Object `s` is created on the stack.
    p = &s;      // t1: Alias `p` is made to refer to object `s`.
  }              // t2: The lifetime of `s` ends. `p` now dangles.
  *p = 42;       // t3: The dangling alias `p` is used. This is a use-after-free.
}
```

The fundamental problem is that the alias `p` outlived the object `s` it referred to. The challenge for a static analysis is therefore clear: to prevent temporal safety errors, the compiler must be able to track aliases and understand the lifetime of the objects they refer to. It needs to know the "points-to" set for every alias at every point in the program and verify that, at the moment of use, the alias does not point to an object whose lifetime has ended.

This alias-based perspective is powerful because it generalizes beautifully. The "end of lifetime" event at `t2` doesn't have to be a variable going out of scope. It could be:

*   A call to `delete`, which ends the lifetime of a heap object.
*   A function `return`, which ends the lifetime of all its local variables.
*   A container modification, like `std::vector::push_back()`, which may reallocate storage, ending the lifetime of the objects in the old buffer and invalidating all existing iterators (aliases).

By focusing on tracking aliases and their validity, we can build a unified model to detect a wide range of temporal safety errors without imposing the heavy "aliasing XOR mutability" restrictions of a traditional borrow checker ([source](https://gist.github.com/nmsmith/cdaa94aa74e8e0611221e65db8e41f7b?content_ref=the+major+advancement+is+to+eliminate+the+aliasing+xor+mutability+restriction+amongst+references+and+replace+it+with+a+similar+restriction+applied+to+lifetime+parameters)). This provides a more intuitive and C++-idiomatic path to memory safety.

## Relation with Thread safety

This analysis does not address Thread Safety. Thread safety is concerned with data races that occur across multiple threads. While it is possible to create temporal safety issues in multi-threaded scenarios, this analysis is focused on the sequential lifetime of objects within a single function.

## Quest for Safer Aliasing

Is it possible to achieve memory safety without a restrictive model like Rust's borrow checker? We believe the answer is yes. The key is to shift our focus from *restricting aliases* to *understanding them*. Instead of forbidding programs that have aliased mutable pointers, we can build a model that understands what each pointer can point to at any given time. This approach, similar to the one proposed in P1179 for C++ and explored in modern lifetime systems like Mojo's, allows us to directly detect the root cause of the problem: using a pointer after its target has ceased to exist ([source](http://docs.google.com/document/d/19vbfAiV1yQu3xSMRWjyPUdzyB_LDdVUcKat_HWI1l3g?content_ref=this+approach+similar+to+the+one+proposed+in+p1179+for+c+and+explored+in+modern+lifetime+systems+like+mojo+s+allows+us+to+directly+detect+the+root+cause+of+the+problem+using+a+pointer+after+its+target+has+ceased+to+exist)).

This paper proposes such a model for C++. Let's begin with a simple, yet illustrative, dangling pointer bug:

```cpp
// Example 1: A simple use-after-free
void definite_simple_case() {
  MyObj* p;
  {
    MyObj s;
    p = &s;     // 'p' now points to 's'
  }             // 's' is destroyed, 'p' is now dangling
  (void)*p;     // Use-after-free
}
```

How can a compiler understand that the use of `p` is an error? It needs to answer a series of questions:

1.  What does `p` point to?
2.  When does the object `p` points to cease to be valid?
3.  Is `p` used after that point?

Our model is designed to answer precisely these questions.

## Core Concepts

Our model is built on a few core concepts that allow us to formally track the relationships between pointers and the data they point to.

### Access Paths

An **Access Path** is a symbolic representation of a storage location in the program ([source](https://raw.githubusercontent.com/llvm/llvm-project/0e7c1732a9a7d28549fe5d690083daeb0e5de6b2/clang/lib/Analysis/LifetimeSafety.cpp?content_ref=struct+accesspath+const+clang+valuedecl+d+accesspath+const+clang+valuedecl+d+d+d)). It provides a way to uniquely identify a variable or a sub-object. For now, we will consider simple paths that refer to top-level variables, but the model can be extended to include field accesses (`a.b`), array elements (`a[i]`), and pointer indirections (`p->field`).

### Loans: The Act of Borrowing

A **Loan** is created whenever a reference or pointer to an object is created. It represents the act of "borrowing" that object's storage location ([source](https://raw.githubusercontent.com/llvm/llvm-project/0e7c1732a9a7d28549fe5d690083daeb0e5de6b2/clang/lib/Analysis/LifetimeSafety.cpp?content_ref=information+about+a+single+borrow+or+loan+a+loan+is+created+when+a+reference+or+pointer+is+created)). Each loan is associated with a unique ID and the `AccessPath` of the object being borrowed.

In our `definite_simple_case` example, the expression `&s` creates a loan. The `AccessPath` for this loan is the variable `s`.

### Origins: The Provenance of a Pointer

An **Origin** is a symbolic identifier that represents the *set of possible loans* a pointer-like object could hold at any given time ([source](http://docs.google.com/document/d/1JpJ3M9yeXX-BnC4oKXBvRWzxoFrwziN1RzI4DrMrSp8?content_ref=ime+is+a+symbolic+identifier+representing+a+set+of+loans+from+which+a+pointer+or+reference+could+have+originated)). Every pointer-like variable or expression in the program is associated with an origin.

*   A variable declaration like `MyObj* p` introduces an origin for `p`.
*   An expression like `&s` also has an origin.
*   The complexity of origins can grow with type complexity. For example:
    *   `int* p;` has a single origin.
    *   `int** p;` has two origins: one for the outer pointer and one for the inner pointer. This allows us to distinguish between `p` itself being modified and what `*p` points to being modified.
    *   `struct S { int* p; };` also has an origin associated with the member `p`.

The central goal of our analysis is to determine, for each origin at each point in the program, which loans it might contain.

## Subtyping Rules and Subset Constraints

The relationships between origins are established through the program's semantics, particularly assignments. When a pointer is assigned to another, as in `p = q`, the set of loans that `q` holds must be a subset of the loans that `p` can now hold. This is a fundamental subtyping rule: for `T*'a` to be a subtype of `T*'b`, the set of loans represented by `'a` must be a subset of the loans represented by `'b`.

This leads to the concept of **subset constraints**. An assignment `p = q` generates a constraint `Origin(q) ⊆ Origin(p)`. The analysis doesn't solve these as a global system of equations. Instead, as we will see, it propagates the *consequences* of these constraints—the loans themselves—through the control-flow graph. This is a key departure from the Polonius model, which focuses on propagating the constraints (`'a: 'b`) themselves.

## Invalidations: When Loans Expire

A loan expires when the object it refers to is no longer valid. In our model, this is an **invalidation** event. The most common invalidation is deallocation, which in C++ can mean:
*   A stack variable going out of scope.
*   A `delete` call on a heap-allocated object.
*   A container modification that reallocates its internal storage.

## An Event-Based Representation of the Function

To analyze a function, we first transform its CFG into a sequence of atomic, lifetime-relevant **Events**, which we call **Facts**. These facts abstract away the complexities of C++ syntax and provide a clean input for our analysis. The main facts are:

*   `Issue(LoanID, OriginID)`: A new loan is created. For example, `&s` generates an `Issue` fact.
*   `Expire(LoanID)`: A loan expires. This is generated at the end of a variable's scope.
*   `OriginFlow(Dest, Src, Kill)`: Loans from a source origin flow to a destination origin, as in an assignment. `Kill` indicates whether the destination's old loans are cleared.
*   `Use(OriginID)`: An origin is used, such as in a pointer dereference.

Let's trace our `definite_simple_case` example with these facts:

```cpp
void definite_simple_case() {
  MyObj* p; // Origin for p is O_p
  {
    MyObj s;
    // The expression `&s` generates:
    //   - IssueFact(L1, O_&s)  (A new loan L1 on 's' is created)
    // The assignment `p = &s` generates:
    //   - OriginFlowFact(O_p, O_&s, Kill=true)
    p = &s;
  } // The end of the scope for 's' generates:
    //   - ExpireFact(L1)
  // The dereference `*p` generates:
  //   - UseFact(O_p)
  (void)*p;
}
```

## Flow-Sensitive Lifetime Policy

With the program represented as a stream of facts, we can now define a flow-sensitive policy to answer our three core questions. We do this by maintaining a map from `Origin` to `Set<Loan>` at each program point. This map represents the state of our analysis.

The analysis proceeds as follows:
1.  **Forward Propagation of Loans:** We perform a forward dataflow analysis.
    *   When we encounter an `Issue` fact, we add the new loan to its origin's loan set.
    *   When we see an `OriginFlow` fact, we update the destination origin's loan set with the loans from the source.
    *   At control-flow merge points, we take the *union* of the loan sets from all incoming branches.

2.  **Backward Propagation of Liveness:** We then perform a backward dataflow analysis, starting from `Use` facts.
    *   A `Use` of an origin marks it as "live."
    *   This liveness information is propagated backward. If an origin `O_p` is live, and it received its loans from `O_q`, then `O_q` is also considered live at that point.

3.  **Error Detection:** An error is flagged when the analysis determines that a **live** origin contains a loan that has **expired**.

In our `definite_simple_case` example:
*   The forward analysis determines that at the point of use, `Origin(p)` contains `Loan(s)`.
*   The backward analysis determines that at the point where `s` is destroyed, `Origin(p)` is live.
*   The `ExpireFact` for `Loan(s)` occurs before the `UseFact`.
*   The combination of these three conditions triggers a use-after-free error.

## Without Functions, Our Work is Done Here!

The model described so far works perfectly for a single, monolithic function. However, the moment we introduce function calls, the problem becomes more complex. How do we reason about lifetimes across function boundaries, especially when we can't see the implementation of the called function?

### Effects of a Function Call

A function call has inputs and outputs. From a lifetime perspective, the key challenge is to understand how the lifetimes of the outputs relate to the lifetimes of the inputs.

### Outlives Constraints and Placeholder Origins

When analyzing a function like `const char* get_prefix(const string& s, int len)`, we don't know the specific lifetime of the `s` that will be passed by the caller. To handle this, we introduce **placeholder origins** for the input parameters. These placeholders act as variables in our analysis.

If a function returns a pointer or reference, its lifetime must be tied to one of its inputs. This is an **outlives constraint**. For example, the return value of `get_prefix` must "outlive" the input `s`. In our model, this means the origin of the return value will contain the placeholder loan associated with `s`.

### Opaque Functions

What if a function's implementation is not visible (e.g., it's in a separate translation unit), and it has no lifetime annotations? In this case, we must be conservative. If we pass a pointer to an opaque function, we have to assume it might have been invalidated. Our model handles this by associating a special **OPAQUE loan** with the pointer after the call, signifying that its lifetime is now unknown.

## Why a Borrow Checker is Not the Right Fit for C++

The "aliasing XOR mutability" rule, while powerful, is fundamentally at odds with many idiomatic C++ patterns.
*   **Observer Patterns:** It's common to have multiple non-owning pointers observing a mutable object.
*   **Intrusive Data Structures:** Data structures like intrusive linked lists require objects to hold pointers to one another, creating cycles that are difficult for a traditional borrow checker to handle.
*   **Iterator Invalidation:** The core problem in C++ is often not aliasing itself, but the fact that a mutation can invalidate an alias (e.g., resizing a vector). An alias-based analysis, like the one proposed here, directly models this problem, whereas a borrow checker can feel like an indirect and overly restrictive solution.

By focusing on tracking what pointers can point to, our model avoids rejecting these safe and useful patterns, making it a more natural fit for the existing C++ ecosystem.

## Open Questions

*   **When and if to introduce the term "lifetime"?** The term "lifetime" is heavily associated with Rust's model. This paper has intentionally focused on "Origins" and "Loans" to avoid confusion. Is there a point where introducing "lifetime" would be helpful, or should we stick to the new terminology?
*   **Syntax for Annotations:** While this model is designed to work with minimal annotations, some will be necessary for complex cases. What should the syntax for these annotations look like? Can we build on existing attributes like `[[clang::lifetimebound]]`?
