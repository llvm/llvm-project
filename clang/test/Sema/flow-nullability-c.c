// Consolidated C tests for flow-sensitive nullability analysis. Covers basic
// narrowing, C-specific patterns (nested structs, restrict, compound literals,
// flexible array members, malloc/free, container_of, goto cleanup), C idioms
// (macros, callbacks, errno), and call invalidation semantics.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -Wno-nullable-to-nonnull-conversion -std=c11 %s -verify

typedef __SIZE_TYPE__ size_t;
typedef _Bool bool;
#define true 1
#define false 0
#define NULL ((void *)0)
#define offsetof(type, member) __builtin_offsetof(type, member)

//===----------------------------------------------------------------------===//
// Shared declarations
//===----------------------------------------------------------------------===//

struct Point {
    int x;
    int y;
};

struct Line {
    struct Point * _Nullable start;
    struct Point * _Nullable end;
};

struct Node {
    int value;
    struct Node * _Nullable next;
    struct Node * _Nullable prev;
};

struct Buffer {
    char * _Nullable data;
    size_t len;
    size_t cap;
};

struct Point * _Nullable getPoint(void);
struct Node * _Nullable getNode(void);
int getInt(void);

// Simulated stdlib declarations
void * _Nullable malloc(size_t);
void * _Nullable calloc(size_t, size_t);
void * _Nullable realloc(void * _Nullable, size_t);
void free(void * _Nullable);
void abort(void) __attribute__((noreturn));
void exit(int) __attribute__((noreturn));

//===----------------------------------------------------------------------===//
// Basic narrowing
//===----------------------------------------------------------------------===//

void test_basic_star_deref_warns(int *p) {
    *p = 42; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void test_basic_star_after_check(int *p) {
    if (p) {
        *p = 42; // OK - narrowed
    }
}

void test_basic_arrow_deref_warns(struct Node *p) {
    p->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void test_basic_arrow_after_check(struct Node *p) {
    if (p) {
        p->value = 1; // OK
    }
}

void test_basic_early_return(struct Node *p) {
    if (!p) return;
    p->value = 1; // OK - narrowed by early return
}

void test_basic_null_comparison(struct Node *p) {
    if (p != 0) {
        p->value = 1; // OK
    }
}

void test_basic_linked_list(struct Node * _Nullable head) {
    for (struct Node * _Nullable p = head; p; p = p->next) {
        p->value = 0; // OK
    }
}

//===----------------------------------------------------------------------===//
// Nested struct pointer access
//===----------------------------------------------------------------------===//

void test_nested_struct(struct Line * _Nullable line) {
    if (line && line->start) {
        line->start->x = 1; // OK - both narrowed
    }
}

void test_nested_not_checked(struct Line * _Nonnull line) {
    line->start->x = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

//===----------------------------------------------------------------------===//
// Double-linked list traversal
//===----------------------------------------------------------------------===//

void test_doubly_linked(struct Node * _Nullable head) {
    for (struct Node * _Nullable p = head; p; p = p->next) {
        p->value = 0; // OK - narrowed by loop condition
        if (p->prev) {
            p->prev->value = -1; // OK - narrowed
        }
    }
}

void test_reverse_traversal(struct Node * _Nullable tail) {
    struct Node * _Nullable p = tail;
    while (p) {
        p->value = 0; // OK
        p = p->prev;
    }
}

//===----------------------------------------------------------------------===//
// restrict pointer
//===----------------------------------------------------------------------===//

void test_restrict(int * restrict p) {
    *p = 42; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void test_restrict_checked(int * restrict _Nullable p) {
    if (p)
        *p = 42; // OK
}

//===----------------------------------------------------------------------===//
// Compound literal
//===----------------------------------------------------------------------===//

void test_compound_literal(void) {
    int *p = &(int){42};
    *p = 0; // OK - address-of compound literal is nonnull
}

//===----------------------------------------------------------------------===//
// Designated initializer
//===----------------------------------------------------------------------===//

void test_designated_init(void) {
    struct Point pt = {.x = 1, .y = 2};
    struct Point *pp = &pt;
    pp->x = 3; // OK - address-of
}

//===----------------------------------------------------------------------===//
// Array of pointers
//===----------------------------------------------------------------------===//

void test_pointer_array(struct Node * _Nullable * _Nonnull nodes, int n) {
    for (int i = 0; i < n; i++) {
        struct Node * _Nullable node = nodes[i];
        if (node) {
            node->value = i; // OK - narrowed via local variable
        }
    }
}

//===----------------------------------------------------------------------===//
// Multiple sequential checks
//===----------------------------------------------------------------------===//

void test_sequential_checks(struct Node * _Nullable a,
                            struct Node * _Nullable b,
                            struct Node * _Nullable c) {
    if (!a) return;
    if (!b) return;
    if (!c) return;
    a->value = b->value + c->value; // OK - all narrowed
}

//===----------------------------------------------------------------------===//
// Null check with comparison operators
//===----------------------------------------------------------------------===//

void test_comparison_styles(struct Node *p) {
    if (p != 0) {
        p->value = 1; // OK
    }
}

void test_comparison_null_macro(struct Node *p) {
    if (p != ((void*)0)) {
        p->value = 1; // OK
    }
}

//===----------------------------------------------------------------------===//
// Function returning _Nonnull
//===----------------------------------------------------------------------===//

struct Node * _Nonnull createNode(void);

void test_nonnull_return(void) {
    struct Node *n = createNode();
    n->value = 1; // OK - _Nonnull return
}

//===----------------------------------------------------------------------===//
// Void pointer cast patterns
//===----------------------------------------------------------------------===//

void test_void_ptr_cast(void * _Nonnull raw) {
    struct Node *n = (struct Node *)raw;
    n->value = 1; // OK - nonnull source
}

void test_void_ptr_nullable(void * _Nullable raw) {
    struct Node *n = (struct Node *)raw;
    n->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

//===----------------------------------------------------------------------===//
// Conditional operator
//===----------------------------------------------------------------------===//

void test_cond_op(struct Node * _Nullable p, struct Node * _Nullable q) {
    struct Node *r = p ? p : q;
    if (r)
        r->value = 1; // OK - narrowed
}

//===----------------------------------------------------------------------===//
// Nested conditionals
//===----------------------------------------------------------------------===//

void test_nested_cond(struct Node * _Nullable p) {
    if (p) {
        if (p->next) {
            if (p->next->next) {
                p->next->next->value = 0; // OK - all narrowed
            }
        }
    }
}

//===----------------------------------------------------------------------===//
// Goto-based cleanup pattern
//===----------------------------------------------------------------------===//

int test_goto_cleanup(struct Node * _Nullable p) {
    int result = -1;
    if (!p) goto out;
    result = p->value; // OK - narrowed
out:
    return result;
}

//===----------------------------------------------------------------------===//
// Switch with null check in cases
//===----------------------------------------------------------------------===//

void test_switch_null_check(struct Node * _Nullable p, int choice) {
    switch (choice) {
    case 0:
        if (p)
            p->value = 0; // OK
        break;
    case 1:
        if (!p) return;
        p->value = 1; // OK
        break;
    default:
        break;
    }
}

//===----------------------------------------------------------------------===//
// Comma operator
//===----------------------------------------------------------------------===//

void test_comma(struct Node * _Nullable p) {
    if (!p) return;
    (void)p->value; // OK - narrowed
}

//===----------------------------------------------------------------------===//
// sizeof does not evaluate
//===----------------------------------------------------------------------===//

void test_sizeof_unevaluated(struct Node * _Nullable p) {
    int s = sizeof(p->value); // OK - sizeof is unevaluated
    (void)s;
}

//===----------------------------------------------------------------------===//
// Pointer subtraction
//===----------------------------------------------------------------------===//

void test_ptr_subtraction(int *a, int *b) {
    long diff = a - b; // expected-warning 2{{pointer arithmetic on nullable pointer}} expected-note 2{{add a null check before performing arithmetic}}
    (void)diff;
}

//===----------------------------------------------------------------------===//
// Macro-heavy null-check patterns
//===----------------------------------------------------------------------===//

#define CHECK_NULL(ptr) do { if (!(ptr)) return; } while(0)
#define CHECK_NULL_RET(ptr, ret) do { if (!(ptr)) return (ret); } while(0)
#define ASSERT_NONNULL(ptr) do { if (!(ptr)) abort(); } while(0)
#define DEREF(p) ((p)->value)
#define SAFE_DEREF(p, fallback) ((p) ? (p)->value : (fallback))

void test_check_null_macro(struct Node *p) {
    CHECK_NULL(p);
    p->value = 1; // OK - macro expanded to if(!p) return
}

int test_check_null_ret_macro(struct Node *p) {
    CHECK_NULL_RET(p, -1);
    return p->value; // OK
}

void test_assert_nonnull_macro(struct Node *p) {
    ASSERT_NONNULL(p);
    p->value = 1; // OK - abort() is noreturn
}

void test_deref_macro(struct Node *p) {
    int v = DEREF(p); // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    (void)v;
}

void test_deref_macro_guarded(struct Node *p) {
    if (p) {
        int v = DEREF(p); // OK - narrowed before macro
        (void)v;
    }
}

void test_safe_deref_macro(struct Node *p) {
    int v = SAFE_DEREF(p, -1); // OK - ternary checks p
    (void)v;
}

//===----------------------------------------------------------------------===//
// malloc/free patterns
//===----------------------------------------------------------------------===//

void test_malloc_no_check(void) {
    struct Node * _Nullable n = malloc(sizeof(struct Node));
    n->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    free(n);
}

void test_malloc_checked(void) {
    struct Node *n = (struct Node *)malloc(sizeof(struct Node));
    if (!n) return;
    n->value = 1; // OK - narrowed
    n->next = NULL;
    free(n);
}

void test_malloc_abort(void) {
    struct Node *n = (struct Node *)malloc(sizeof(struct Node));
    if (!n) abort();
    n->value = 1; // OK - abort is noreturn
    free(n);
}

void test_calloc_checked(void) {
    struct Node *n = (struct Node *)calloc(1, sizeof(struct Node));
    if (!n) return;
    n->value = 1; // OK
    free(n);
}

//===----------------------------------------------------------------------===//
// realloc pattern
//===----------------------------------------------------------------------===//

void test_realloc(struct Buffer * _Nonnull buf) {
    char * _Nullable new_data = (char *)realloc(buf->data, buf->cap * 2);
    if (!new_data) return;
    buf->data = new_data;
    buf->cap *= 2;
}

//===----------------------------------------------------------------------===//
// Linked list construction and traversal
//===----------------------------------------------------------------------===//

struct Node * _Nullable list_prepend(struct Node * _Nullable head, int val) {
    struct Node *n = (struct Node *)malloc(sizeof(struct Node));
    if (!n) return head;
    n->value = val; // OK - checked
    n->next = head;
    return n;
}

void list_free(struct Node * _Nullable head) {
    struct Node * _Nullable p = head;
    while (p) {
        struct Node * _Nullable next_node = p->next; // OK - p narrowed
        free(p);
        p = next_node;
    }
}

int list_sum(struct Node * _Nullable head) {
    int sum = 0;
    for (struct Node * _Nullable p = head; p; p = p->next) {
        sum += p->value; // OK - narrowed by loop condition
    }
    return sum;
}

//===----------------------------------------------------------------------===//
// Callback / function pointer patterns
//===----------------------------------------------------------------------===//

typedef void (*node_visitor_fn)(struct Node * _Nonnull, void * _Nullable);

void list_foreach(struct Node * _Nullable head, node_visitor_fn _Nonnull fn, void * _Nullable ctx) {
    for (struct Node * _Nullable p = head; p; p = p->next) {
        fn(p, ctx); // OK - p narrowed
    }
}

//===----------------------------------------------------------------------===//
// errno-style error checking
//===----------------------------------------------------------------------===//

struct File;
struct File * _Nullable file_open(const char * _Nonnull path);
int file_read(struct File * _Nonnull f, char * _Nonnull buf, int len);
void file_close(struct File * _Nonnull f);

int test_errno_pattern(void) {
    struct File * _Nullable f = file_open("/tmp/test");
    if (!f) return -1;
    char buf[256];
    int n = file_read(f, buf, 256); // OK
    file_close(f); // OK
    return n;
}

//===----------------------------------------------------------------------===//
// container_of macro pattern
//===----------------------------------------------------------------------===//

#define container_of(ptr, type, member) \
    ((type *)((char *)(ptr) - offsetof(type, member)))

struct list_head {
    struct list_head * _Nullable next;
    struct list_head * _Nullable prev;
};

struct my_item {
    int data;
    struct list_head link;
};

void test_container_of(struct list_head * _Nullable pos) {
    if (!pos) return;
    struct my_item *item = container_of(pos, struct my_item, link);
    item->data = 42; // OK - arithmetic on non-null pointer
}

//===----------------------------------------------------------------------===//
// Multi-level goto cleanup
//===----------------------------------------------------------------------===//

int test_multi_level_cleanup(void) {
    int ret = -1;
    struct Node *a = (struct Node *)malloc(sizeof(struct Node));
    if (!a) goto out;

    struct Node *b = (struct Node *)malloc(sizeof(struct Node));
    if (!b) goto free_a;

    a->value = 1; // OK - narrowed past goto
    b->value = 2; // OK - narrowed past goto
    a->next = b;
    ret = a->value + b->value;

free_a:
    free(a);
out:
    return ret;
}

//===----------------------------------------------------------------------===//
// Bitfield struct with nullable pointer
//===----------------------------------------------------------------------===//

struct Options {
    unsigned verbose : 1;
    unsigned debug : 1;
    struct Node * _Nullable config;
};

void test_bitfield_struct(struct Options * _Nonnull opts) {
    if (opts->config) {
        opts->config->value = opts->verbose; // OK - narrowed
    }
}

//===----------------------------------------------------------------------===//
// Null check via helper function (intraprocedural limitation)
//===----------------------------------------------------------------------===//

static bool is_valid(const struct Node * _Nullable p) {
    return p != NULL;
}

void test_helper_check(struct Node *p) {
    // The analysis can't see inside helper functions -- accepted limitation.
    if (is_valid(p)) {
        p->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

//===----------------------------------------------------------------------===//
// __builtin_expect / LIKELY / UNLIKELY macros
//===----------------------------------------------------------------------===//

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

void test_likely_macro(struct Node *p) {
    if (LIKELY(p != NULL)) {
        p->value = 1; // OK
    }
}

void test_unlikely_null(struct Node *p) {
    if (UNLIKELY(p == NULL)) return;
    p->value = 1; // OK
}

//===----------------------------------------------------------------------===//
// Flexible array member
//===----------------------------------------------------------------------===//

struct FlexArray {
    int count;
    struct Node * _Nullable items[];
};

void test_flex_array(struct FlexArray * _Nonnull fa) {
    for (int i = 0; i < fa->count; i++) {
        struct Node * _Nullable item = fa->items[i];
        if (item) {
            item->value = i; // OK
        }
    }
}

//===----------------------------------------------------------------------===//
// void** output parameter pattern
//===----------------------------------------------------------------------===//

int get_node_out(struct Node * _Nullable * _Nonnull out);

void test_output_param(void) {
    struct Node * _Nullable n = NULL;
    if (get_node_out(&n) == 0 && n) {
        n->value = 42; // OK - checked via &&
    }
}

//===----------------------------------------------------------------------===//
// Static assert + null check
//===----------------------------------------------------------------------===//

_Static_assert(sizeof(struct Node) > 0, "Node must have size");

void test_with_static_assert(struct Node *p) {
    _Static_assert(sizeof(*p) == sizeof(struct Node), "size match");
    if (p) {
        p->value = 1; // OK
    }
}

//===----------------------------------------------------------------------===//
// Call invalidation: function calls do NOT invalidate narrowing
//===----------------------------------------------------------------------===//
// Functions receive a copy of pointer arguments, so they cannot modify
// the original pointer variable to make it null.

void takes_int(int x);
void takes_ptr(int *p);

void test_narrowing_preserved_after_call(int *p) {
    if (p) {
        takes_int(42);
        *p = 1; // OK - p is still nonnull
    }
}

void test_narrowing_preserved_pass_ptr(int *p) {
    if (p) {
        takes_ptr(p);
        *p = 1; // OK - pass by value
    }
}

void test_multiple_calls(int *p, int *q) {
    if (p && q) {
        takes_ptr(p);
        takes_ptr(q);
        takes_int(42);
        *p = 1; // OK - narrowing preserved through all calls
        *q = 2; // OK
    }
}

// Known false negative: passing a pointer's address lets the callee
// set *out = NULL, invalidating narrowing. The analysis intentionally
// does not invalidate on address-taken (matching ThreadSafety's approach).
void nullify(int **out);

void test_address_taken_false_negative(int *p) {
    if (p) {
        nullify(&p);
        *p = 1; // no warning - known false negative
    }
}
