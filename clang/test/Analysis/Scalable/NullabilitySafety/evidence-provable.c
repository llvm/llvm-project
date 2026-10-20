// Nonnull evidence must be proven, not assumed. Under
// -fnullability-default=nonnull an unannotated parameter, field or call result
// is merely treated as nonnull; passing or returning one says nothing about the
// value. A null test on a parameter, a field, or a local copied from either
// (directly or through a chain of copies) rules out _Nonnull, except inside an
// assert, which claims the opposite. The parameters of a function whose address
// is taken have callers the analysis cannot see, so they are never _Nonnull
// either.

// RUN: rm -f %t.json
// RUN: %clang_cc1 -fsyntax-only -fnullability-default=nonnull %s \
// RUN:   --ssaf-extract-summaries=NullabilitySafety \
// RUN:   --ssaf-compilation-unit-id=tu --ssaf-tu-summary-file=%t.json
// RUN: %python %S/Inputs/decode-summary.py %t.json | FileCheck %s

#define assert(x) ((x) ? (void)0 : __builtin_trap())

struct Node { struct Node *next; };
struct Db { struct Node *head; void *arg; };

void take_addr(int *p);
void take_call(int *p);
void take_deref(int *p);
void take_node(struct Node *n);
void take_arg(void *a);
int *make(void);
void cb(void *ctx, int *p);
void (*handler)(void *, int *) = cb;

void from_field(struct Db *db) {
  take_arg(db->arg);
  if (db->head)
    take_node(db->head);
}

void from_param(int *p) {
  take_deref(p);
  if (p == 0)
    return;
}

void loop_copy(struct Node *head) {
  struct Node *n;
  for (n = head; n; n = n->next)
    ;
}

int list_walk(struct Db *db) {
  int n = 0;
  for (struct Node *p = db->head->next; p; p = p->next)
    n++;
  return n;
}

void proven(int *p) {
  static int x;
  take_addr(&x);
  take_call(make());
  assert(p != 0);
}

int *returns_field(struct Db *db) { return (int *)db->arg; }

int *returns_param(int *p) {
  static int x;
  if (!p)
    return &x;
  return p;
}

void copy_chain(int *p) {
  int *a = p;
  int *b = a;
  if (!b)
    return;
}
void copy_cycle(int *p) {
  int *a = p, *b;
  b = a;
  a = b;
  if (a == 0)
    return;
}
void unused_ctx(void) {
  static int x;
  cb(0, &x);
}

// CHECK-NOT: {{.}}
// CHECK:      c:@F@copy_chain MaybeNullEvidence c:@F@copy_chain param 1
// CHECK-NEXT: c:@F@copy_cycle MaybeNullEvidence c:@F@copy_cycle param 1
// CHECK-NEXT: c:@F@from_field ConditionalEvidence c:@F@take_arg param 1 <- c:@S@Db@FI@arg
// CHECK-NEXT: c:@F@from_field MaybeNullEvidence c:@S@Db@FI@head
// CHECK-NEXT: c:@F@from_field NonnullEvidence c:@F@take_node param 1
// CHECK-NEXT: c:@F@from_param ConditionalEvidence c:@F@take_deref param 1 <- c:@F@from_param param 1
// CHECK-NEXT: c:@F@from_param MaybeNullEvidence c:@F@from_param param 1
// CHECK-NEXT: c:@F@list_walk MaybeNullEvidence c:@S@Node@FI@next
// CHECK-NEXT: c:@F@loop_copy MaybeNullEvidence c:@F@loop_copy param 1
// CHECK-NEXT: c:@F@loop_copy MaybeNullEvidence c:@S@Node@FI@next
// CHECK-NEXT: c:@F@proven ConditionalEvidence c:@F@take_call param 1 <- c:@F@make return
// CHECK-NEXT: c:@F@proven NonnullEvidence c:@F@take_addr param 1
// CHECK-NEXT: c:@F@returns_field ConditionalEvidence c:@F@returns_field return <- c:@S@Db@FI@arg
// CHECK-NEXT: c:@F@returns_param AllReturnsNonnull c:@F@returns_param return
// CHECK-NEXT: c:@F@returns_param MaybeNullEvidence c:@F@returns_param param 1
// CHECK-NEXT: c:@F@returns_param NonnullEvidence c:@F@returns_param return
// CHECK-NEXT: c:@F@unused_ctx NonnullEvidence c:@F@cb param 2
// CHECK-NEXT: c:@F@unused_ctx NullableEvidence c:@F@cb param 1
// CHECK-NEXT: c:@handler MaybeNullEvidence c:@F@cb param 1
// CHECK-NEXT: c:@handler MaybeNullEvidence c:@F@cb param 2
// CHECK-NOT: {{.}}
