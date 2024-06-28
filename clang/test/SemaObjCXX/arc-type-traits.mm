// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -fobjc-runtime-has-weak -verify -std=c++11 %s -Wno-deprecated-builtins
// expected-no-diagnostics

// Check the results of the various type-trait query functions on
// lifetime-qualified types in ARC.

struct HasStrong { id obj; };
struct HasWeak { __weak id obj; };
struct HasUnsafeUnretained { __unsafe_unretained id obj; };

// __has_nothrow_assign
static_assert(__has_nothrow_assign(__strong id), "");
static_assert(__has_nothrow_assign(__weak id), "");
static_assert(__has_nothrow_assign(__autoreleasing id), "");
static_assert(__has_nothrow_assign(__unsafe_unretained id), "");
static_assert(__has_nothrow_assign(HasStrong), "");
static_assert(__has_nothrow_assign(HasWeak), "");
static_assert(__has_nothrow_assign(HasUnsafeUnretained), "");

// __has_nothrow_copy
static_assert(__has_nothrow_copy(__strong id), "");
static_assert(__has_nothrow_copy(__weak id), "");
static_assert(__has_nothrow_copy(__autoreleasing id), "");
static_assert(__has_nothrow_copy(__unsafe_unretained id), "");
static_assert(__has_nothrow_copy(HasStrong), "");
static_assert(__has_nothrow_copy(HasWeak), "");
static_assert(__has_nothrow_copy(HasUnsafeUnretained), "");

// __has_nothrow_constructor
static_assert(__has_nothrow_constructor(__strong id), "");
static_assert(__has_nothrow_constructor(__weak id), "");
static_assert(__has_nothrow_constructor(__autoreleasing id), "");
static_assert(__has_nothrow_constructor(__unsafe_unretained id), "");
static_assert(__has_nothrow_constructor(HasStrong), "");
static_assert(__has_nothrow_constructor(HasWeak), "");
static_assert(__has_nothrow_constructor(HasUnsafeUnretained), "");

// __has_trivial_assign
static_assert(!__has_trivial_assign(__strong id), "");
static_assert(!__has_trivial_assign(__weak id), "");
static_assert(!__has_trivial_assign(__autoreleasing id), "");
static_assert(__has_trivial_assign(__unsafe_unretained id), "");
static_assert(!__has_trivial_assign(HasStrong), "");
static_assert(!__has_trivial_assign(HasWeak), "");
static_assert(__has_trivial_assign(HasUnsafeUnretained), "");

// __has_trivial_copy
static_assert(!__has_trivial_copy(__strong id), "");
static_assert(!__has_trivial_copy(__weak id), "");
static_assert(!__has_trivial_copy(__autoreleasing id), "");
static_assert(__has_trivial_copy(__unsafe_unretained id), "");
static_assert(!__has_trivial_copy(HasStrong), "");
static_assert(!__has_trivial_copy(HasWeak), "");
static_assert(__has_trivial_copy(HasUnsafeUnretained), "");

// __has_trivial_constructor
static_assert(!__has_trivial_constructor(__strong id), "");
static_assert(!__has_trivial_constructor(__weak id), "");
static_assert(!__has_trivial_constructor(__autoreleasing id), "");
static_assert(__has_trivial_constructor(__unsafe_unretained id), "");
static_assert(!__has_trivial_constructor(HasStrong), "");
static_assert(!__has_trivial_constructor(HasWeak), "");
static_assert(__has_trivial_constructor(HasUnsafeUnretained), "");

// __has_trivial_destructor
static_assert(!__has_trivial_destructor(__strong id), "");
static_assert(!__has_trivial_destructor(__weak id), "");
static_assert(__has_trivial_destructor(__autoreleasing id), "");
static_assert(__has_trivial_destructor(__unsafe_unretained id), "");
static_assert(!__has_trivial_destructor(HasStrong), "");
static_assert(!__has_trivial_destructor(HasWeak), "");
static_assert(__has_trivial_destructor(HasUnsafeUnretained), "");

// __is_literal
static_assert(__is_literal(__strong id), "");
static_assert(__is_literal(__weak id), "");
static_assert(__is_literal(__autoreleasing id), "");
static_assert(__is_literal(__unsafe_unretained id), "");

// __is_literal_type
static_assert(__is_literal_type(__strong id), "");
static_assert(__is_literal_type(__weak id), "");
static_assert(__is_literal_type(__autoreleasing id), "");
static_assert(__is_literal_type(__unsafe_unretained id), "");

// __is_pod
static_assert(!__is_pod(__strong id), "");
static_assert(!__is_pod(__weak id), "");
static_assert(!__is_pod(__autoreleasing id), "");
static_assert(__is_pod(__unsafe_unretained id), "");
static_assert(!__is_pod(HasStrong), "");
static_assert(!__is_pod(HasWeak), "");
static_assert(__is_pod(HasUnsafeUnretained), "");

// __is_trivial
static_assert(!__is_trivial(__strong id), "");
static_assert(!__is_trivial(__weak id), "");
static_assert(!__is_trivial(__autoreleasing id), "");
static_assert(__is_trivial(__unsafe_unretained id), "");
static_assert(!__is_trivial(HasStrong), "");
static_assert(!__is_trivial(HasWeak), "");
static_assert(__is_trivial(HasUnsafeUnretained), "");

// __is_scalar
static_assert(!__is_scalar(__strong id), "");
static_assert(!__is_scalar(__weak id), "");
static_assert(!__is_scalar(__autoreleasing id), "");
static_assert(__is_scalar(__unsafe_unretained id), "");

// __is_standard_layout
static_assert(__is_standard_layout(__strong id), "");
static_assert(__is_standard_layout(__weak id), "");
static_assert(__is_standard_layout(__autoreleasing id), "");
static_assert(__is_standard_layout(__unsafe_unretained id), "");

// __is_trivally_assignable
static_assert(!__is_trivially_assignable(__strong id&, __strong id), "");
static_assert(!__is_trivially_assignable(__strong id&, __weak id), "");
static_assert(!__is_trivially_assignable(__strong id&, __autoreleasing id), "");
static_assert(!__is_trivially_assignable(__strong id&, __unsafe_unretained id), "");
static_assert(!__is_trivially_assignable(__strong id&, __strong id&&), "");
static_assert(!__is_trivially_assignable(__strong id&, __weak id&&), "");
static_assert(!__is_trivially_assignable(__strong id&, __autoreleasing id&&), "");
static_assert(!__is_trivially_assignable(__strong id&, __unsafe_unretained id&&), "");
static_assert(!__is_trivially_assignable(__weak id&, __strong id), "");
static_assert(!__is_trivially_assignable(__weak id&, __weak id), "");
static_assert(!__is_trivially_assignable(__weak id&, __autoreleasing id), "");
static_assert(!__is_trivially_assignable(__weak id&, __unsafe_unretained id), "");
static_assert(!__is_trivially_assignable(__weak id&, __strong id&&), "");
static_assert(!__is_trivially_assignable(__weak id&, __weak id&&), "");
static_assert(!__is_trivially_assignable(__weak id&, __autoreleasing id&&), "");
static_assert(!__is_trivially_assignable(__weak id&, __unsafe_unretained id&&), "");

static_assert(!__is_trivially_assignable(__autoreleasing id&, __strong id), "");
static_assert(!__is_trivially_assignable(__autoreleasing id&, __weak id), "");
static_assert(!__is_trivially_assignable(__autoreleasing id&, __autoreleasing id), "");
static_assert(!__is_trivially_assignable(__autoreleasing id&, __unsafe_unretained id), "");
static_assert(!__is_trivially_assignable(__autoreleasing id&, __strong id&&), "");
static_assert(!__is_trivially_assignable(__autoreleasing id&, __weak id&&), "");
static_assert(!__is_trivially_assignable(__autoreleasing id&, __autoreleasing id&&), "");
static_assert(!__is_trivially_assignable(__autoreleasing id&, __unsafe_unretained id&&), "");

static_assert(__is_trivially_assignable(__unsafe_unretained id&, __strong id), "");
static_assert(__is_trivially_assignable(__unsafe_unretained id&, __weak id), "");
static_assert(__is_trivially_assignable(__unsafe_unretained id&, __autoreleasing id), "");
static_assert(__is_trivially_assignable(__unsafe_unretained id&, __unsafe_unretained id), "");
static_assert(__is_trivially_assignable(__unsafe_unretained id&, __strong id&&), "");
static_assert(__is_trivially_assignable(__unsafe_unretained id&, __weak id&&), "");
static_assert(__is_trivially_assignable(__unsafe_unretained id&, __autoreleasing id&&), "");
static_assert(__is_trivially_assignable(__unsafe_unretained id&, __unsafe_unretained id&&), "");

static_assert(!__is_trivially_assignable(HasStrong&, HasStrong), "");
static_assert(!__is_trivially_assignable(HasStrong&, HasStrong&&), "");
static_assert(!__is_trivially_assignable(HasWeak&, HasWeak), "");
static_assert(!__is_trivially_assignable(HasWeak&, HasWeak&&), "");
static_assert(__is_trivially_assignable(HasUnsafeUnretained&, HasUnsafeUnretained), "");
static_assert(__is_trivially_assignable(HasUnsafeUnretained&, HasUnsafeUnretained&&), "");

// __is_trivally_constructible
static_assert(!__is_trivially_constructible(__strong id), "");
static_assert(!__is_trivially_constructible(__weak id), "");
static_assert(!__is_trivially_constructible(__autoreleasing id), "");
static_assert(__is_trivially_constructible(__unsafe_unretained id), "");

static_assert(!__is_trivially_constructible(__strong id, __strong id), "");
static_assert(!__is_trivially_constructible(__strong id, __weak id), "");
static_assert(!__is_trivially_constructible(__strong id, __autoreleasing id), "");
static_assert(!__is_trivially_constructible(__strong id, __unsafe_unretained id), "");
static_assert(!__is_trivially_constructible(__strong id, __strong id&&), "");
static_assert(!__is_trivially_constructible(__strong id, __weak id&&), "");
static_assert(!__is_trivially_constructible(__strong id, __autoreleasing id&&), "");
static_assert(!__is_trivially_constructible(__strong id, __unsafe_unretained id&&), "");
static_assert(!__is_trivially_constructible(__weak id, __strong id), "");
static_assert(!__is_trivially_constructible(__weak id, __weak id), "");
static_assert(!__is_trivially_constructible(__weak id, __autoreleasing id), "");
static_assert(!__is_trivially_constructible(__weak id, __unsafe_unretained id), "");
static_assert(!__is_trivially_constructible(__weak id, __strong id&&), "");
static_assert(!__is_trivially_constructible(__weak id, __weak id&&), "");
static_assert(!__is_trivially_constructible(__weak id, __autoreleasing id&&), "");
static_assert(!__is_trivially_constructible(__weak id, __unsafe_unretained id&&), "");

static_assert(!__is_trivially_constructible(__autoreleasing id, __strong id), "");
static_assert(!__is_trivially_constructible(__autoreleasing id, __weak id), "");
static_assert(!__is_trivially_constructible(__autoreleasing id, __autoreleasing id), "");
static_assert(!__is_trivially_constructible(__autoreleasing id, __unsafe_unretained id), "");
static_assert(!__is_trivially_constructible(__autoreleasing id, __strong id&&), "");
static_assert(!__is_trivially_constructible(__autoreleasing id, __weak id&&), "");
static_assert(!__is_trivially_constructible(__autoreleasing id, __autoreleasing id&&), "");
static_assert(!__is_trivially_constructible(__autoreleasing id, __unsafe_unretained id&&), "");

static_assert(__is_trivially_constructible(__unsafe_unretained id, __strong id), "");
static_assert(__is_trivially_constructible(__unsafe_unretained id, __weak id), "");
static_assert(__is_trivially_constructible(__unsafe_unretained id, __autoreleasing id), "");
static_assert(__is_trivially_constructible(__unsafe_unretained id, __unsafe_unretained id), "");
static_assert(__is_trivially_constructible(__unsafe_unretained id, __strong id&&), "");
static_assert(__is_trivially_constructible(__unsafe_unretained id, __weak id&&), "");
static_assert(__is_trivially_constructible(__unsafe_unretained id, __autoreleasing id&&), "");
static_assert(__is_trivially_constructible(__unsafe_unretained id, __unsafe_unretained id&&), "");

static_assert(!__is_trivially_constructible(HasStrong, HasStrong), "");
static_assert(!__is_trivially_constructible(HasStrong, HasStrong&&), "");
static_assert(!__is_trivially_constructible(HasWeak, HasWeak), "");
static_assert(!__is_trivially_constructible(HasWeak, HasWeak&&), "");
static_assert(__is_trivially_constructible(HasUnsafeUnretained, HasUnsafeUnretained), "");
static_assert(__is_trivially_constructible(HasUnsafeUnretained, HasUnsafeUnretained&&), "");

// __is_trivially_relocatable
static_assert(__is_trivially_relocatable(__strong id), "");
static_assert(!__is_trivially_relocatable(__weak id), "");
static_assert(__is_trivially_relocatable(__autoreleasing id), "");
static_assert(__is_trivially_relocatable(__unsafe_unretained id), "");
static_assert(__is_trivially_relocatable(HasStrong), "");
static_assert(!__is_trivially_relocatable(HasWeak), "");
static_assert(__is_trivially_relocatable(HasUnsafeUnretained), "");

// __is_bitwise_cloneable
static_assert(!__is_bitwise_cloneable(__strong id), "");
static_assert(!__is_bitwise_cloneable(__weak id), "");
static_assert(!__is_bitwise_cloneable(__autoreleasing id), "");
static_assert(__is_bitwise_cloneable(__unsafe_unretained id), "");
static_assert(!__is_bitwise_cloneable(HasStrong), "");
static_assert(!__is_bitwise_cloneable(HasWeak), "");
static_assert(__is_bitwise_cloneable(HasUnsafeUnretained), "");
