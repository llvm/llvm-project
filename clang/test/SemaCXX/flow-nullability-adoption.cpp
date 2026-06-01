// Consolidated adoption, configuration, and meta tests for flow-sensitive
// nullability analysis. Covers gradual adoption gating,
// false-positive suppression, type identity preservation, and performance
// stress patterns.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 %s -verify
// UNSUPPORTED: asan, msan, ubsan

//===----------------------------------------------------------------------===//
// Shared declarations
//===----------------------------------------------------------------------===//

struct Node {
    int value;
    Node * _Nullable next;
    Node * _Nullable left;
    Node * _Nullable right;
};

struct Entity {
    int x;
    int value() const { return x; }
};

Node * _Nullable getNode();
Entity * _Nullable getHead();
Entity * _Nullable getChest();
Entity *getEntityUnannotated();
int getInt();

//===----------------------------------------------------------------------===//
// Gradual adoption: per-function gating
//===----------------------------------------------------------------------===//
// Adapted from gradual-adoption tests. Under -fnullability-default=nullable,
// all pointers default to nullable so analysis is always active. We wrap tests
// in assume_nonnull to mirror how real code opts into the analysis.

// Outside any pragma, with nullable default, unannotated params ARE nullable.
void test_adoption_outside_pragma_explicit_nullable(Entity * _Nullable p) {
    p->x = 1;              // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    (*p).x = 1;            // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    getHead()->x = 1;      // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

#pragma clang assume_nonnull begin

void test_adoption_explicit_nullable_arrow(Entity * _Nullable p) {
    p->x = 1;              // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void test_adoption_explicit_nullable_arrow_checked(Entity * _Nullable p) {
    if (!p) return;
    p->x = 1;              // OK - narrowed to nonnull
}

void test_adoption_explicit_nullable_star(Entity * _Nullable p) {
    (*p).x = 1;            // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void test_adoption_explicit_nullable_star_checked(Entity * _Nullable p) {
    if (!p) return;
    (*p).x = 1;            // OK - narrowed to nonnull
}

void test_adoption_chained_nullable_arrow() {
    getHead()->x = 1;      // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void test_adoption_chained_nullable_method() {
    int v = getHead()->value(); // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// getEntityUnannotated() is declared outside assume_nonnull, so under
// -fnullability-default=nullable its return type is nullable. Under
// -fnullability-default=unspecified (the original gradual-adoption test),
// it would NOT warn. This function validates nullable-mode behavior.
Entity * _Nonnull getEntityNonnull();

void test_adoption_unannotated_no_warn() {
    // Use a _Nonnull-declared function to mirror the assume_nonnull behavior
    Entity *e = getEntityNonnull();
    e->x = 1;              // OK - nonnull return
    (*e).x = 1;            // OK
}

void test_adoption_unannotated_param_no_warn(Entity *p) {
    p->x = 1;              // OK - nonnull via pragma
}

// Lambda scoping: analysis must still run for outer function.
// Regression test: lambda bodies call ActOnStartOfFunctionDef, which must
// not clobber the per-function analysis decision for the enclosing function.

void test_adoption_lambda_no_clobber(Entity * _Nullable p) {
    auto f = [](int x) { return x + 1; };
    (void)f(1);
    p->x = 1;              // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void test_adoption_nested_lambda_scoping(Entity * _Nullable p) {
    auto outer = [](int x) {
        auto inner = [](int y) { return y; };
        return inner(x);
    };
    (void)outer(1);
    (*p).x = 1;            // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

#pragma clang assume_nonnull end


//===----------------------------------------------------------------------===//
// False-positive regression suite
//===----------------------------------------------------------------------===//
// Every test case in this section must produce NO warnings. These represent
// common C++ patterns that an overly-aggressive analysis might flag.

#pragma clang assume_nonnull begin

// --- Conditional initialization on all paths ---

void test_fp_conditional_init(bool cond) {
    int x = 0, y = 0;
    int *p;
    if (cond) {
        p = &x;
    } else {
        p = &y;
    }
    (void)*p; // OK - assigned nonnull on both paths
}

// --- Static local variable ---

void test_fp_static_local() {
    static int x = 42;
    int *p = &x;
    (void)*p; // OK - address-of is always nonnull
}

// --- Global variable access ---

int g_fp_value = 0;

void test_fp_global_addr() {
    int *p = &g_fp_value;
    (void)*p; // OK - address-of
}

// --- Function pointer call ---

typedef int (*IntFn)(int);

void test_fp_fn_ptr(IntFn fn) {
    int result = fn(42); // OK - no * dereference
}

// --- Chained method calls on nonnull ---

struct Builder {
    Builder *setX(int) { return this; }
    Builder *setY(int) { return this; }
    int build() { return 0; }
};

void test_fp_builder_pattern() {
    Builder b;
    b.setX(1)->setY(2)->build(); // OK - this is nonnull
}

// --- Address-of array element ---

void test_fp_array_element_addr() {
    int arr[10];
    int *p = &arr[5];
    (void)*p; // OK - address-of
}

// --- Pointer to member of stack object ---

void test_fp_member_addr() {
    Node n;
    int *p = &n.value;
    (void)*p; // OK - address-of
}

// --- Ternary with nonnull on both sides ---

void test_fp_ternary_both_nonnull(bool cond) {
    int x = 1, y = 2;
    int *p = cond ? &x : &y;
    (void)*p; // OK - nonnull on both branches
}

// --- Cast of nonnull ---

void test_fp_cast_nonnull() {
    int x = 42;
    void *vp = &x;
    int *ip = static_cast<int *>(vp);
    (void)*ip; // OK - source was nonnull (address-of)
}

// --- new expression ---

void test_fp_throwing_new() {
    int *p = new int(42);
    (void)*p; // OK - throwing new never returns null
}

// --- Multiple checks, then use ---

void test_fp_multi_check(Node * _Nullable a, Node * _Nullable b, Node * _Nullable c) {
    if (a && b && c) {
        (void)a->value; // OK
        (void)b->value; // OK
        (void)c->value; // OK
    }
}

// --- Reassign to nonnull after nullable ---

void test_fp_reassign_nonnull() {
    int x;
    int * _Nullable p = nullptr;
    p = &x;
    (void)*p; // OK - reassigned to nonnull
}

// --- Loop variable always nonnull ---

void test_fp_loop_var() {
    int arr[10];
    for (int i = 0; i < 10; i++) {
        int *p = &arr[i];
        (void)*p; // OK - address-of
    }
}

// --- Nested struct access on nonnull ---

struct Outer {
    Node node;
};

void test_fp_nested_nonnull_access() {
    Outer o;
    int v = o.node.value; // OK - dot access on stack object
}

// --- Pointer arithmetic on nonnull ---

void test_fp_ptr_arith() {
    int arr[10];
    int *p = arr;
    int *q = arr + 5;
    (void)*q; // OK
}

// --- Reference binding ---

void test_fp_reference(int * _Nonnull p) {
    int &ref = *p; // OK - _Nonnull
    ref = 42;
}

// --- Comma operator with pointer ---

void test_fp_comma_op() {
    int x;
    int *p = (getInt(), &x);
    (void)*p; // OK - comma evaluates to &x which is nonnull
}

// --- Narrowing survives function calls ---

void fp_external_fn();

void test_fp_narrowing_survives_call(Node * _Nullable p) {
    if (!p) return;
    fp_external_fn();
    (void)p->value; // OK - function call doesn't invalidate narrowing
}

// --- Member narrowing in array subscripts and across calls ---

struct MemberNarrowObj {
    int * _Nullable arr;
    int * _Nullable other;
    int getMember() const;

    void test_member_subscript_after_check() {
        if (!arr) return;
        (void)arr[0]; // OK - member narrowed by null check
    }

    void test_member_subscript_across_call() {
        if (!arr) return;
        getMember();
        (void)arr[0]; // OK - calls don't invalidate member narrowing
    }

    void test_member_subscript_in_loop() {
        if (!arr) return;
        for (int i = 0; i < 3; i++) {
            (void)arr[i]; // OK - member narrowed before loop
        }
    }

    void test_member_deref_across_call() {
        if (!arr) return;
        getMember();
        (void)*arr; // OK - calls don't invalidate member narrowing
    }

    void test_member_subscript_no_check() {
        (void)arr[0]; // expected-warning {{dereference of nullable pointer}} \
                      // expected-note {{add a null check}}
    }

    void test_two_members_narrowed() {
        if (!arr || !other) return;
        (void)arr[0]; // OK
        (void)other[0]; // OK
    }

    // Explicit == nullptr checks, disjunctions, if-body narrowing

    void test_member_eq_nullptr_early_return() {
        if (arr == nullptr) return;
        (void)arr[0]; // OK - == nullptr style check
    }

    void test_member_ne_nullptr_if_body() {
        // Narrowing inside if-body (not early return)
        if (arr != nullptr) {
            (void)arr[0]; // OK - narrowed inside if-body
        }
    }

    void test_member_disjunction_guard() {
        // Multiple members checked with || and early return
        if (arr == nullptr || other == nullptr) return;
        (void)arr[0]; // OK - both narrowed after disjunction
        (void)other[0]; // OK
    }

    void test_member_subscript_in_loop_with_calls() {
        if (arr == nullptr || other == nullptr) return;
        for (int i = 0; i < 3; i++) {
            (void)arr[i]; // OK - narrowing survives loop back-edge
            other[i] = getMember(); // OK - call + subscript in loop
        }
    }

    void test_member_ne_nullptr_if_body_with_loop() {
        // if (ptr != nullptr) { loop { ptr[i] } }
        if (arr != nullptr) {
            for (int i = 0; i < 3; i++) {
                (void)arr[i]; // OK - narrowed inside if-body, survives loop
            }
        }
    }

    void test_member_subscript_with_fn_index() {
        // ptr[getIndex()] after null check
        if (arr == nullptr) return;
        (void)arr[getMember()]; // OK - function call as index doesn't affect base narrowing
    }

    // Passing narrowed member to _Nonnull parameter
    static void accept_nonnull(int * _Nonnull p);
    void test_member_nonnull_arg_after_check() {
        if (arr == nullptr) return;
        accept_nonnull(arr); // OK - member narrowed, safe to pass to _Nonnull
    }

    void test_member_nonnull_arg_no_check() {
        accept_nonnull(arr); // expected-warning {{nullable pointer}} \
                             // expected-note {{add a null check}}
    }
};

// --- sizeof/alignof don't dereference ---

void test_fp_sizeof_no_deref(Node * _Nullable p) {
    auto s = sizeof(*p); // OK - sizeof doesn't evaluate its operand
    (void)s;
}

// --- decltype doesn't dereference ---

void test_fp_decltype_no_deref(Node * _Nullable p) {
    using T = decltype(p->value); // OK - decltype is unevaluated
    T x = 0;
    (void)x;
}

// --- this pointer in member functions ---

struct FPObj {
    int x;
    void method() {
        this->x = 1; // OK - this is never null
        (*this).x = 2; // OK - *this is suppressed
    }
};

// --- Non-std iterator dereference ---

struct FPIterator {
    Node *current;
    Node &operator*() { return *current; }
    Node *operator->() { return current; }
};

void test_fp_iterator_deref(FPIterator it) {
    (void)it->value; // OK - non-std operator-> is not checked
}

#pragma clang assume_nonnull end

//===----------------------------------------------------------------------===//
// Type identity: nullability must not affect type system
//===----------------------------------------------------------------------===//
// Nullability qualifiers are type sugar in Clang -- they don't participate
// in template argument deduction, std::is_same, decltype, or overload
// resolution.

template<typename T, typename U>
struct is_same { static constexpr bool value = false; };
template<typename T>
struct is_same<T, T> { static constexpr bool value = true; };

#pragma clang assume_nonnull begin

void test_ti_decltype_local() {
    int x = 0;
    int *p = &x;
    static_assert(is_same<decltype(p), int*>::value, "");
}

void test_ti_decltype_param(int *p) {
    static_assert(is_same<decltype(p), int*>::value, "");
}

void test_ti_auto_deduction() {
    int x = 0;
    auto p = &x;
    static_assert(is_same<decltype(p), int*>::value, "");
}

template<typename T>
void ti_accept(T) {
    static_assert(is_same<T, int*>::value, "");
}

void test_ti_template_deduction() {
    int x = 0;
    int *p = &x;
    ti_accept(p);
}

template<typename T> void ti_accept_ptr(T*) {}

void test_ti_explicit_template_arg() {
    int x = 0;
    int *p = &x;
    ti_accept_ptr<int>(p);
}

void test_ti_nullability_is_sugar() {
    int x;
    int *bare = &x;
    int * _Nullable nullable = &x;
    int * _Nonnull nonnull = &x;
    int * _Null_unspecified unspec = &x;

    static_assert(is_same<decltype(bare), decltype(nullable)>::value, "");
    static_assert(is_same<decltype(bare), decltype(nonnull)>::value, "");
    static_assert(is_same<decltype(bare), decltype(unspec)>::value, "");
}

auto ti_make_ptr() {
    int *p = new int(42);
    return p;
}

void test_ti_return_type_deduction() {
    static_assert(is_same<decltype(ti_make_ptr()), int*>::value, "");
}

void test_ti_const_ptr() {
    const int x = 0;
    const int *p = &x;
    static_assert(is_same<decltype(p), const int*>::value, "");
}

void test_ti_ptr_to_ptr() {
    int x;
    int *p = &x;
    int **pp = &p;
    static_assert(is_same<decltype(pp), int**>::value, "");
}

#pragma clang assume_nonnull end

//===----------------------------------------------------------------------===//
// Performance stress tests
//===----------------------------------------------------------------------===//
// Generates a large amount of work for the analysis. Must compile within the
// default lit timeout. If the analysis has a complexity regression, this test
// will time out. Modeled after clang/test/Analysis/runtime-regression.c.

#pragma clang assume_nonnull begin

// --- Pattern 1: Many sequential null checks (tests linear scaling) ---

#define CHECK_AND_USE(N) \
    { Node * _Nullable p##N = getNode(); if (p##N) p##N->value = N; }

void stress_sequential() {
    CHECK_AND_USE(0)  CHECK_AND_USE(1)  CHECK_AND_USE(2)  CHECK_AND_USE(3)
    CHECK_AND_USE(4)  CHECK_AND_USE(5)  CHECK_AND_USE(6)  CHECK_AND_USE(7)
    CHECK_AND_USE(8)  CHECK_AND_USE(9)  CHECK_AND_USE(10) CHECK_AND_USE(11)
    CHECK_AND_USE(12) CHECK_AND_USE(13) CHECK_AND_USE(14) CHECK_AND_USE(15)
    CHECK_AND_USE(16) CHECK_AND_USE(17) CHECK_AND_USE(18) CHECK_AND_USE(19)
    CHECK_AND_USE(20) CHECK_AND_USE(21) CHECK_AND_USE(22) CHECK_AND_USE(23)
    CHECK_AND_USE(24) CHECK_AND_USE(25) CHECK_AND_USE(26) CHECK_AND_USE(27)
    CHECK_AND_USE(28) CHECK_AND_USE(29) CHECK_AND_USE(30) CHECK_AND_USE(31)
    CHECK_AND_USE(32) CHECK_AND_USE(33) CHECK_AND_USE(34) CHECK_AND_USE(35)
    CHECK_AND_USE(36) CHECK_AND_USE(37) CHECK_AND_USE(38) CHECK_AND_USE(39)
    CHECK_AND_USE(40) CHECK_AND_USE(41) CHECK_AND_USE(42) CHECK_AND_USE(43)
    CHECK_AND_USE(44) CHECK_AND_USE(45) CHECK_AND_USE(46) CHECK_AND_USE(47)
    CHECK_AND_USE(48) CHECK_AND_USE(49) CHECK_AND_USE(50) CHECK_AND_USE(51)
    CHECK_AND_USE(52) CHECK_AND_USE(53) CHECK_AND_USE(54) CHECK_AND_USE(55)
    CHECK_AND_USE(56) CHECK_AND_USE(57) CHECK_AND_USE(58) CHECK_AND_USE(59)
    CHECK_AND_USE(60) CHECK_AND_USE(61) CHECK_AND_USE(62) CHECK_AND_USE(63)
    CHECK_AND_USE(64) CHECK_AND_USE(65) CHECK_AND_USE(66) CHECK_AND_USE(67)
    CHECK_AND_USE(68) CHECK_AND_USE(69) CHECK_AND_USE(70) CHECK_AND_USE(71)
    CHECK_AND_USE(72) CHECK_AND_USE(73) CHECK_AND_USE(74) CHECK_AND_USE(75)
    CHECK_AND_USE(76) CHECK_AND_USE(77) CHECK_AND_USE(78) CHECK_AND_USE(79)
    CHECK_AND_USE(80) CHECK_AND_USE(81) CHECK_AND_USE(82) CHECK_AND_USE(83)
    CHECK_AND_USE(84) CHECK_AND_USE(85) CHECK_AND_USE(86) CHECK_AND_USE(87)
    CHECK_AND_USE(88) CHECK_AND_USE(89) CHECK_AND_USE(90) CHECK_AND_USE(91)
    CHECK_AND_USE(92) CHECK_AND_USE(93) CHECK_AND_USE(94) CHECK_AND_USE(95)
    CHECK_AND_USE(96) CHECK_AND_USE(97) CHECK_AND_USE(98) CHECK_AND_USE(99)
}

// --- Pattern 2: Branch fan-out (tests intersect scaling) ---

#define BRANCH(N) if (getInt()) { s##N = &nodes[N]; }

void stress_fanout() {
    Node nodes[50];
    Node * _Nullable s0 = nullptr, * _Nullable s1 = nullptr;
    Node * _Nullable s2 = nullptr, * _Nullable s3 = nullptr;
    Node * _Nullable s4 = nullptr, * _Nullable s5 = nullptr;
    Node * _Nullable s6 = nullptr, * _Nullable s7 = nullptr;
    Node * _Nullable s8 = nullptr, * _Nullable s9 = nullptr;
    Node * _Nullable s10 = nullptr, * _Nullable s11 = nullptr;
    Node * _Nullable s12 = nullptr, * _Nullable s13 = nullptr;
    Node * _Nullable s14 = nullptr, * _Nullable s15 = nullptr;
    Node * _Nullable s16 = nullptr, * _Nullable s17 = nullptr;
    Node * _Nullable s18 = nullptr, * _Nullable s19 = nullptr;
    Node * _Nullable s20 = nullptr, * _Nullable s21 = nullptr;
    Node * _Nullable s22 = nullptr, * _Nullable s23 = nullptr;
    Node * _Nullable s24 = nullptr, * _Nullable s25 = nullptr;
    Node * _Nullable s26 = nullptr, * _Nullable s27 = nullptr;
    Node * _Nullable s28 = nullptr, * _Nullable s29 = nullptr;
    Node * _Nullable s30 = nullptr, * _Nullable s31 = nullptr;
    Node * _Nullable s32 = nullptr, * _Nullable s33 = nullptr;
    Node * _Nullable s34 = nullptr, * _Nullable s35 = nullptr;
    Node * _Nullable s36 = nullptr, * _Nullable s37 = nullptr;
    Node * _Nullable s38 = nullptr, * _Nullable s39 = nullptr;
    Node * _Nullable s40 = nullptr, * _Nullable s41 = nullptr;
    Node * _Nullable s42 = nullptr, * _Nullable s43 = nullptr;
    Node * _Nullable s44 = nullptr, * _Nullable s45 = nullptr;
    Node * _Nullable s46 = nullptr, * _Nullable s47 = nullptr;
    Node * _Nullable s48 = nullptr, * _Nullable s49 = nullptr;

    BRANCH(0)  BRANCH(1)  BRANCH(2)  BRANCH(3)  BRANCH(4)
    BRANCH(5)  BRANCH(6)  BRANCH(7)  BRANCH(8)  BRANCH(9)
    BRANCH(10) BRANCH(11) BRANCH(12) BRANCH(13) BRANCH(14)
    BRANCH(15) BRANCH(16) BRANCH(17) BRANCH(18) BRANCH(19)
    BRANCH(20) BRANCH(21) BRANCH(22) BRANCH(23) BRANCH(24)
    BRANCH(25) BRANCH(26) BRANCH(27) BRANCH(28) BRANCH(29)
    BRANCH(30) BRANCH(31) BRANCH(32) BRANCH(33) BRANCH(34)
    BRANCH(35) BRANCH(36) BRANCH(37) BRANCH(38) BRANCH(39)
    BRANCH(40) BRANCH(41) BRANCH(42) BRANCH(43) BRANCH(44)
    BRANCH(45) BRANCH(46) BRANCH(47) BRANCH(48) BRANCH(49)
}

// --- Pattern 3: Many small functions (realistic workload) ---

#define SMALL_FN(N) \
    void small_fn_##N(Node * _Nullable p) { \
        if (!p) return; \
        p->value = N; \
        if (p->next) p->next->value = N + 1; \
    }

SMALL_FN(0)  SMALL_FN(1)  SMALL_FN(2)  SMALL_FN(3)  SMALL_FN(4)
SMALL_FN(5)  SMALL_FN(6)  SMALL_FN(7)  SMALL_FN(8)  SMALL_FN(9)
SMALL_FN(10) SMALL_FN(11) SMALL_FN(12) SMALL_FN(13) SMALL_FN(14)
SMALL_FN(15) SMALL_FN(16) SMALL_FN(17) SMALL_FN(18) SMALL_FN(19)
SMALL_FN(20) SMALL_FN(21) SMALL_FN(22) SMALL_FN(23) SMALL_FN(24)
SMALL_FN(25) SMALL_FN(26) SMALL_FN(27) SMALL_FN(28) SMALL_FN(29)
SMALL_FN(30) SMALL_FN(31) SMALL_FN(32) SMALL_FN(33) SMALL_FN(34)
SMALL_FN(35) SMALL_FN(36) SMALL_FN(37) SMALL_FN(38) SMALL_FN(39)
SMALL_FN(40) SMALL_FN(41) SMALL_FN(42) SMALL_FN(43) SMALL_FN(44)
SMALL_FN(45) SMALL_FN(46) SMALL_FN(47) SMALL_FN(48) SMALL_FN(49)
SMALL_FN(50) SMALL_FN(51) SMALL_FN(52) SMALL_FN(53) SMALL_FN(54)
SMALL_FN(55) SMALL_FN(56) SMALL_FN(57) SMALL_FN(58) SMALL_FN(59)
SMALL_FN(60) SMALL_FN(61) SMALL_FN(62) SMALL_FN(63) SMALL_FN(64)
SMALL_FN(65) SMALL_FN(66) SMALL_FN(67) SMALL_FN(68) SMALL_FN(69)
SMALL_FN(70) SMALL_FN(71) SMALL_FN(72) SMALL_FN(73) SMALL_FN(74)
SMALL_FN(75) SMALL_FN(76) SMALL_FN(77) SMALL_FN(78) SMALL_FN(79)
SMALL_FN(80) SMALL_FN(81) SMALL_FN(82) SMALL_FN(83) SMALL_FN(84)
SMALL_FN(85) SMALL_FN(86) SMALL_FN(87) SMALL_FN(88) SMALL_FN(89)
SMALL_FN(90) SMALL_FN(91) SMALL_FN(92) SMALL_FN(93) SMALL_FN(94)
SMALL_FN(95) SMALL_FN(96) SMALL_FN(97) SMALL_FN(98) SMALL_FN(99)

// --- Pattern 4: Deep nesting (tests edge state tracking) ---

void stress_deep_nesting(
    Node * _Nullable p0,  Node * _Nullable p1,  Node * _Nullable p2,
    Node * _Nullable p3,  Node * _Nullable p4,  Node * _Nullable p5,
    Node * _Nullable p6,  Node * _Nullable p7,  Node * _Nullable p8,
    Node * _Nullable p9,  Node * _Nullable p10, Node * _Nullable p11,
    Node * _Nullable p12, Node * _Nullable p13, Node * _Nullable p14) {
    if (p0) {
     if (p1) {
      if (p2) {
       if (p3) {
        if (p4) {
         if (p5) {
          if (p6) {
           if (p7) {
            if (p8) {
             if (p9) {
              if (p10) {
               if (p11) {
                if (p12) {
                 if (p13) {
                  if (p14) {
                    p0->value = p1->value + p2->value + p3->value;
                    p4->value = p5->value + p6->value + p7->value;
                    p8->value = p9->value + p10->value + p11->value;
                    p12->value = p13->value + p14->value;
                  }
                 }
                }
               }
              }
             }
            }
           }
          }
         }
        }
       }
      }
     }
    }
}

// --- Pattern 5: Linked list traversal with operations ---

void stress_linked_list() {
    Node * _Nullable head = getNode();
    int sum = 0;
    for (Node * _Nullable p = head; p; p = p->next) {
        sum += p->value;
        if (p->left) {
            sum += p->left->value;
            if (p->left->right) {
                sum += p->left->right->value;
            }
        }
        if (p->right) {
            sum += p->right->value;
        }
    }
    (void)sum;
}

// --- Pattern 6: Diamond CFG merges ---

#define DIAMOND(N) \
    if (getInt()) { \
        if (p##N) p##N->value = N; \
    } else { \
        if (q##N) q##N->value = N; \
    }

void stress_diamond_merges() {
    Node * _Nullable p0 = getNode(), * _Nullable q0 = getNode();
    Node * _Nullable p1 = getNode(), * _Nullable q1 = getNode();
    Node * _Nullable p2 = getNode(), * _Nullable q2 = getNode();
    Node * _Nullable p3 = getNode(), * _Nullable q3 = getNode();
    Node * _Nullable p4 = getNode(), * _Nullable q4 = getNode();
    Node * _Nullable p5 = getNode(), * _Nullable q5 = getNode();
    Node * _Nullable p6 = getNode(), * _Nullable q6 = getNode();
    Node * _Nullable p7 = getNode(), * _Nullable q7 = getNode();
    Node * _Nullable p8 = getNode(), * _Nullable q8 = getNode();
    Node * _Nullable p9 = getNode(), * _Nullable q9 = getNode();
    Node * _Nullable p10 = getNode(), * _Nullable q10 = getNode();
    Node * _Nullable p11 = getNode(), * _Nullable q11 = getNode();
    Node * _Nullable p12 = getNode(), * _Nullable q12 = getNode();
    Node * _Nullable p13 = getNode(), * _Nullable q13 = getNode();
    Node * _Nullable p14 = getNode(), * _Nullable q14 = getNode();
    Node * _Nullable p15 = getNode(), * _Nullable q15 = getNode();
    Node * _Nullable p16 = getNode(), * _Nullable q16 = getNode();
    Node * _Nullable p17 = getNode(), * _Nullable q17 = getNode();
    Node * _Nullable p18 = getNode(), * _Nullable q18 = getNode();
    Node * _Nullable p19 = getNode(), * _Nullable q19 = getNode();
    Node * _Nullable p20 = getNode(), * _Nullable q20 = getNode();
    Node * _Nullable p21 = getNode(), * _Nullable q21 = getNode();
    Node * _Nullable p22 = getNode(), * _Nullable q22 = getNode();
    Node * _Nullable p23 = getNode(), * _Nullable q23 = getNode();
    Node * _Nullable p24 = getNode(), * _Nullable q24 = getNode();

    DIAMOND(0)  DIAMOND(1)  DIAMOND(2)  DIAMOND(3)  DIAMOND(4)
    DIAMOND(5)  DIAMOND(6)  DIAMOND(7)  DIAMOND(8)  DIAMOND(9)
    DIAMOND(10) DIAMOND(11) DIAMOND(12) DIAMOND(13) DIAMOND(14)
    DIAMOND(15) DIAMOND(16) DIAMOND(17) DIAMOND(18) DIAMOND(19)
    DIAMOND(20) DIAMOND(21) DIAMOND(22) DIAMOND(23) DIAMOND(24)
}

// --- Pattern 7: Boolean guard stress ---

#define BOOL_GUARD(N) \
    bool valid_##N = (getNode() != nullptr); \
    Node * _Nullable bg_##N = getNode();

#define BOOL_CHECK(N) \
    if (valid_##N && bg_##N) { bg_##N->value = N; }

void stress_bool_guards() {
    BOOL_GUARD(0)  BOOL_GUARD(1)  BOOL_GUARD(2)  BOOL_GUARD(3)
    BOOL_GUARD(4)  BOOL_GUARD(5)  BOOL_GUARD(6)  BOOL_GUARD(7)
    BOOL_GUARD(8)  BOOL_GUARD(9)  BOOL_GUARD(10) BOOL_GUARD(11)
    BOOL_GUARD(12) BOOL_GUARD(13) BOOL_GUARD(14) BOOL_GUARD(15)
    BOOL_GUARD(16) BOOL_GUARD(17) BOOL_GUARD(18) BOOL_GUARD(19)
    BOOL_GUARD(20) BOOL_GUARD(21) BOOL_GUARD(22) BOOL_GUARD(23)
    BOOL_GUARD(24) BOOL_GUARD(25) BOOL_GUARD(26) BOOL_GUARD(27)
    BOOL_GUARD(28) BOOL_GUARD(29) BOOL_GUARD(30) BOOL_GUARD(31)
    BOOL_GUARD(32) BOOL_GUARD(33) BOOL_GUARD(34) BOOL_GUARD(35)
    BOOL_GUARD(36) BOOL_GUARD(37) BOOL_GUARD(38) BOOL_GUARD(39)

    BOOL_CHECK(0)  BOOL_CHECK(1)  BOOL_CHECK(2)  BOOL_CHECK(3)
    BOOL_CHECK(4)  BOOL_CHECK(5)  BOOL_CHECK(6)  BOOL_CHECK(7)
    BOOL_CHECK(8)  BOOL_CHECK(9)  BOOL_CHECK(10) BOOL_CHECK(11)
    BOOL_CHECK(12) BOOL_CHECK(13) BOOL_CHECK(14) BOOL_CHECK(15)
    BOOL_CHECK(16) BOOL_CHECK(17) BOOL_CHECK(18) BOOL_CHECK(19)
    BOOL_CHECK(20) BOOL_CHECK(21) BOOL_CHECK(22) BOOL_CHECK(23)
    BOOL_CHECK(24) BOOL_CHECK(25) BOOL_CHECK(26) BOOL_CHECK(27)
    BOOL_CHECK(28) BOOL_CHECK(29) BOOL_CHECK(30) BOOL_CHECK(31)
    BOOL_CHECK(32) BOOL_CHECK(33) BOOL_CHECK(34) BOOL_CHECK(35)
    BOOL_CHECK(36) BOOL_CHECK(37) BOOL_CHECK(38) BOOL_CHECK(39)
}

// --- Pattern 8: Member narrowing stress ---

struct Tree {
    int data;
    Tree * _Nullable left;
    Tree * _Nullable right;
    Tree * _Nullable parent;
};

#define MEMBER_NARROW(N) \
    void member_fn_##N(Tree * _Nullable t) { \
        if (!t) return; \
        if (t->left) { \
            t->left->data = N; \
            if (t->left->right) t->left->right->data = N; \
        } \
        if (t->right) { \
            t->right->data = N; \
            if (t->right->parent) t->right->parent->data = N; \
        } \
    }

MEMBER_NARROW(0)  MEMBER_NARROW(1)  MEMBER_NARROW(2)  MEMBER_NARROW(3)
MEMBER_NARROW(4)  MEMBER_NARROW(5)  MEMBER_NARROW(6)  MEMBER_NARROW(7)
MEMBER_NARROW(8)  MEMBER_NARROW(9)  MEMBER_NARROW(10) MEMBER_NARROW(11)
MEMBER_NARROW(12) MEMBER_NARROW(13) MEMBER_NARROW(14) MEMBER_NARROW(15)
MEMBER_NARROW(16) MEMBER_NARROW(17) MEMBER_NARROW(18) MEMBER_NARROW(19)
MEMBER_NARROW(20) MEMBER_NARROW(21) MEMBER_NARROW(22) MEMBER_NARROW(23)
MEMBER_NARROW(24) MEMBER_NARROW(25) MEMBER_NARROW(26) MEMBER_NARROW(27)
MEMBER_NARROW(28) MEMBER_NARROW(29) MEMBER_NARROW(30) MEMBER_NARROW(31)
MEMBER_NARROW(32) MEMBER_NARROW(33) MEMBER_NARROW(34) MEMBER_NARROW(35)
MEMBER_NARROW(36) MEMBER_NARROW(37) MEMBER_NARROW(38) MEMBER_NARROW(39)
MEMBER_NARROW(40) MEMBER_NARROW(41) MEMBER_NARROW(42) MEMBER_NARROW(43)
MEMBER_NARROW(44) MEMBER_NARROW(45) MEMBER_NARROW(46) MEMBER_NARROW(47)
MEMBER_NARROW(48) MEMBER_NARROW(49)

// --- Pattern 9: Compound conditions stress ---

#define AND_CHAIN_3(A, B, C) if (A && B && C) { A->value = B->value + C->value; }

void stress_compound_conditions() {
    Node * _Nullable a0 = getNode(), * _Nullable b0 = getNode(), * _Nullable c0 = getNode();
    Node * _Nullable a1 = getNode(), * _Nullable b1 = getNode(), * _Nullable c1 = getNode();
    Node * _Nullable a2 = getNode(), * _Nullable b2 = getNode(), * _Nullable c2 = getNode();
    Node * _Nullable a3 = getNode(), * _Nullable b3 = getNode(), * _Nullable c3 = getNode();
    Node * _Nullable a4 = getNode(), * _Nullable b4 = getNode(), * _Nullable c4 = getNode();
    Node * _Nullable a5 = getNode(), * _Nullable b5 = getNode(), * _Nullable c5 = getNode();
    Node * _Nullable a6 = getNode(), * _Nullable b6 = getNode(), * _Nullable c6 = getNode();
    Node * _Nullable a7 = getNode(), * _Nullable b7 = getNode(), * _Nullable c7 = getNode();
    Node * _Nullable a8 = getNode(), * _Nullable b8 = getNode(), * _Nullable c8 = getNode();
    Node * _Nullable a9 = getNode(), * _Nullable b9 = getNode(), * _Nullable c9 = getNode();

    AND_CHAIN_3(a0, b0, c0) AND_CHAIN_3(a1, b1, c1)
    AND_CHAIN_3(a2, b2, c2) AND_CHAIN_3(a3, b3, c3)
    AND_CHAIN_3(a4, b4, c4) AND_CHAIN_3(a5, b5, c5)
    AND_CHAIN_3(a6, b6, c6) AND_CHAIN_3(a7, b7, c7)
    AND_CHAIN_3(a8, b8, c8) AND_CHAIN_3(a9, b9, c9)
}

// --- Pattern 10: Large switch statement ---

void stress_switch() {
    Node * _Nullable p = getNode();
    int x = getInt();
    switch (x) {
    case 0:  if (p) p->value = 0;  break;
    case 1:  if (p) p->value = 1;  break;
    case 2:  if (p) p->value = 2;  break;
    case 3:  if (p) p->value = 3;  break;
    case 4:  if (p) p->value = 4;  break;
    case 5:  if (p) p->value = 5;  break;
    case 6:  if (p) p->value = 6;  break;
    case 7:  if (p) p->value = 7;  break;
    case 8:  if (p) p->value = 8;  break;
    case 9:  if (p) p->value = 9;  break;
    case 10: if (p) p->value = 10; break;
    case 11: if (p) p->value = 11; break;
    case 12: if (p) p->value = 12; break;
    case 13: if (p) p->value = 13; break;
    case 14: if (p) p->value = 14; break;
    case 15: if (p) p->value = 15; break;
    case 16: if (p) p->value = 16; break;
    case 17: if (p) p->value = 17; break;
    case 18: if (p) p->value = 18; break;
    case 19: if (p) p->value = 19; break;
    case 20: if (p) p->value = 20; break;
    case 21: if (p) p->value = 21; break;
    case 22: if (p) p->value = 22; break;
    case 23: if (p) p->value = 23; break;
    case 24: if (p) p->value = 24; break;
    case 25: if (p) p->value = 25; break;
    case 26: if (p) p->value = 26; break;
    case 27: if (p) p->value = 27; break;
    case 28: if (p) p->value = 28; break;
    case 29: if (p) p->value = 29; break;
    case 30: if (p) p->value = 30; break;
    case 31: if (p) p->value = 31; break;
    case 32: if (p) p->value = 32; break;
    case 33: if (p) p->value = 33; break;
    case 34: if (p) p->value = 34; break;
    case 35: if (p) p->value = 35; break;
    case 36: if (p) p->value = 36; break;
    case 37: if (p) p->value = 37; break;
    case 38: if (p) p->value = 38; break;
    case 39: if (p) p->value = 39; break;
    case 40: if (p) p->value = 40; break;
    case 41: if (p) p->value = 41; break;
    case 42: if (p) p->value = 42; break;
    case 43: if (p) p->value = 43; break;
    case 44: if (p) p->value = 44; break;
    case 45: if (p) p->value = 45; break;
    case 46: if (p) p->value = 46; break;
    case 47: if (p) p->value = 47; break;
    case 48: if (p) p->value = 48; break;
    case 49: if (p) p->value = 49; break;
    default: break;
    }
}

#pragma clang assume_nonnull end
