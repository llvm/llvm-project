// RUN: %clang_cc1 -fsyntax-only -verify=expected,beta -Wthread-safety -Wthread-safety-beta %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,nobeta -Wthread-safety %s

// Local pointer variables whose value the analysis cannot resolve, e.g.
// because the value has been loaded by inline assembly as in the arm64
// implementation of smp_load_acquire().

struct __attribute__((capability("spinlock"))) spinlock {
  int x;
};
struct wq {
  struct spinlock lock;
  int data __attribute__((guarded_by(lock)));
};
struct req {
  struct wq *head;
};
struct holder {
  struct wq *wq;
};
void spin_lock(struct spinlock *l) __attribute__((acquire_capability(l)));
void spin_unlock(struct spinlock *l) __attribute__((release_capability(l)));
int busy(struct req *r);

#define load_acquire_asm(p)                                                    \
  ({                                                                           \
    union {                                                                    \
      __typeof__(*(p)) __val;                                                  \
      char __c[1];                                                             \
    } __u;                                                                     \
    __typeof__(p) __p = (p);                                                   \
    asm volatile("" : "=r"(*(unsigned long *)__u.__c) : "m"(*__p));            \
    (__typeof__(*(p)))__u.__val;                                               \
  })

// Do not name the capability after a variable that is declared inside the
// statement expression.
void name_of_alias(struct req *r) {
  struct wq *head = load_acquire_asm(&r->head);
  spin_lock(&head->lock); // expected-note{{spinlock acquired here}}
} // expected-warning{{spinlock 'head->lock' is still held at the end of function}}

// Statement expressions appearing as subexpressions of a local pointer
// initializer must also fall back to the local variable name rather than
// collapsing distinct capabilities onto '#undefined'.
void subexpr_stmt_expr_distinct(struct holder **p1, struct holder **p2) {
  struct wq *a = load_acquire_asm(p1)->wq;
  struct wq *b = load_acquire_asm(p2)->wq;
  spin_lock(&a->lock); // expected-note{{spinlock acquired here}}
  b->data = 42;        // expected-warning{{writing variable 'data' requires holding spinlock 'b->lock' exclusively}}
  // expected-note@-1{{found near match 'a->lock'}}
  spin_unlock(&b->lock); // expected-warning{{releasing spinlock 'b->lock' that was not held}}
} // expected-warning{{spinlock 'a->lock' is still held at the end of function}}

void nested_stmt_expr_distinct(struct holder **p1, struct holder **p2) {
  struct wq *a = ({
    struct holder *__h = load_acquire_asm(p1);
    __h->wq;
  });
  struct wq *b = ({
    struct holder *__h = load_acquire_asm(p2);
    __h->wq;
  });
  spin_lock(&a->lock); // expected-note{{spinlock acquired here}}
  spin_unlock(&b->lock); // expected-warning{{releasing spinlock 'b->lock' that was not held}}
} // expected-warning{{spinlock 'a->lock' is still held at the end of function}}

// The same if the address of a local variable of a nested statement
// expression that is not the value of the enclosing statement expression is
// stored in a local variable of the enclosing statement expression.
void nested_non_last_stmt_expr_distinct(struct holder *p1,
                                        struct holder *p2) {
  struct wq *a = ({
    struct holder *__p;
    ({
      struct holder __h = *p1;
      __p = &__h;
    });
    __p->wq;
  });
  struct wq *b = ({
    struct holder *__p;
    ({
      struct holder __h = *p2;
      __p = &__h;
    });
    __p->wq;
  });
  spin_lock(&a->lock); // expected-note{{spinlock acquired here}}
  spin_unlock(&b->lock); // expected-warning{{releasing spinlock 'b->lock' that was not held}}
} // expected-warning{{spinlock 'a->lock' is still held at the end of function}}

// Resolvable local variables declared inside a statement expression are still
// looked through when -Wthread-safety-beta is enabled.
#define passthrough_stmt_expr(x)                                               \
  ({                                                                           \
    __typeof__(x) __x = (x);                                                   \
    __x;                                                                       \
  })

void resolvable_local_in_stmt_expr(struct req *r) {
  struct wq *w = passthrough_stmt_expr(r->head);
  spin_lock(&w->lock); // nobeta-note{{spinlock acquired here}}
  r->head->data = 1; // nobeta-warning{{writing variable 'data' requires holding spinlock 'r->head->lock' exclusively}}
  // nobeta-note@-1{{found near match 'w->lock'}}
  spin_unlock(&r->head->lock); // nobeta-warning{{releasing spinlock 'r->head->lock' that was not held}}
} // nobeta-warning{{spinlock 'w->lock' is still held at the end of function}}

// Variables with static storage duration declared inside a statement
// expression exist outside of it, so they may be used as names.
struct wq global_wq;
void needs_global_wq_lock(void)
    __attribute__((requires_capability(global_wq.lock)));

void extern_decl_in_stmt_expr(void) {
  struct wq *w = ({
    extern struct wq global_wq;
    &global_wq;
  });
  spin_lock(&w->lock);
  global_wq.data = 1; // nobeta-warning{{writing variable 'data' requires holding spinlock 'global_wq.lock' exclusively}}
  // nobeta-note@-1{{found near match 'w->lock'}}
  needs_global_wq_lock(); // nobeta-warning{{calling function 'needs_global_wq_lock' requires holding spinlock 'global_wq.lock' exclusively}}
  // nobeta-note@-1{{found near match 'w->lock'}}
  spin_unlock(&w->lock);
}

void static_local_in_stmt_expr(void) {
  struct wq *w = ({
    static struct wq local_wq;
    &local_wq;
  });
  spin_lock(&w->lock);
  w->data = 1;
  spin_unlock(&w->lock);
}

// 'b' is looked through since 'hp' still has the value it had in the
// definition of 'b', but 'a' is not.
void outer_var_reassigned_to_stmt_expr_local_distinct(struct holder *p1,
                                                      struct holder *p2) {
  struct holder *hp = 0;
  struct wq *a = ({
    struct holder __h = *p1;
    hp = &__h;
    hp->wq;
  });
  struct wq *b = ({
    struct holder __h = *p2;
    hp = &__h;
    hp->wq;
  });
  spin_lock(&a->lock); // expected-note{{spinlock acquired here}}
  b->data = 42;        // beta-warning{{writing variable 'data' requires holding spinlock 'hp->wq->lock' exclusively}} nobeta-warning{{writing variable 'data' requires holding spinlock 'b->lock' exclusively}}
  // expected-note@-1{{found near match 'a->lock'}}
  spin_unlock(&b->lock); // beta-warning{{releasing spinlock 'hp->wq->lock' that was not held}} nobeta-warning{{releasing spinlock 'b->lock' that was not held}}
} // expected-warning{{spinlock 'a->lock' is still held at the end of function}}

// A variable declared in a nested statement expression only makes the
// innermost statement expression unresolvable. 'hp' is declared outside both
// statement expressions, so 'a' can still be looked through.
void nested_stmt_expr_local_discarded(struct holder **p1) {
  struct holder *hp = 0;
  struct wq *a = ({
    hp = load_acquire_asm(p1);
    hp->wq;
  });
  spin_lock(&hp->wq->lock);
  a->data = 1; // nobeta-warning{{writing variable 'data' requires holding spinlock 'a->lock' exclusively}}
  // nobeta-note@-1{{found near match 'hp->wq->lock'}}
  spin_unlock(&hp->wq->lock);
}

// A variable declared outside a statement expression that is assigned a
// variable declared inside it is not named after the latter, neither inside
// nor outside the statement expression.
void outer_var_assigned_stmt_expr_local(struct holder **p1) {
  struct holder *hp = 0;
  struct wq *a = ({
    struct holder *__h = load_acquire_asm(p1);
    hp = __h;
    hp->wq;
  });
  spin_lock(&hp->wq->lock);
  a->data = 1; // nobeta-warning{{writing variable 'data' requires holding spinlock 'a->lock' exclusively}}
  // nobeta-note@-1{{found near match 'hp->wq->lock'}}
  spin_unlock(&hp->wq->lock);
}

void outer_var_assigned_stmt_expr_local_name(struct holder **p1) {
  struct holder *hp = 0;
  ({
    struct holder *__h = load_acquire_asm(p1);
    hp = __h;
  });
  spin_lock(&hp->wq->lock); // expected-note{{spinlock acquired here}}
} // expected-warning{{spinlock 'hp->wq->lock' is still held at the end of function}}

#define load_into_asm(var, p)                                                  \
  ({                                                                           \
    __typeof__(*(p)) __val;                                                    \
    asm volatile("" : "=r"(__val) : "m"(*(p)));                                \
    (var) = __val;                                                             \
  })

void assigned_inside_stmt_expr(struct req *r1, struct req *r2) {
  struct wq *a, *b;
  load_into_asm(a, &r1->head);
  load_into_asm(b, &r2->head);
  spin_lock(&a->lock);
  b->data = 42; // expected-warning{{writing variable 'data' requires holding spinlock 'b->lock' exclusively}}
  // expected-note@-1{{found near match 'a->lock'}}
  spin_unlock(&a->lock);
}
