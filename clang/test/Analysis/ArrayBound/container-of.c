// RUN: %clang_analyze_cc1 -Wno-array-bounds -Wno-address-of-packed-member \
// RUN:   -analyzer-checker=core,security.ArrayBound,unix.Malloc \
// RUN:   -verify %s
//

#define offsetof(TYPE, MEMBER) __builtin_offsetof(TYPE, MEMBER)
#define container_of(PTR, TYPE, MEMBER)                                  \
  ((TYPE *)((char *)(PTR) - offsetof(TYPE, MEMBER)))
#define container_of_uchar(PTR, TYPE, MEMBER)                            \
  ((TYPE *)((unsigned char *)(PTR) - offsetof(TYPE, MEMBER)))
#define container_of_typed(PTR, TYPE, MEMBER) ({                         \
  const __typeof__(((TYPE *)0)->MEMBER) *__member_ptr = (PTR);           \
  (TYPE *)((char *)__member_ptr - offsetof(TYPE, MEMBER));               \
})

void *malloc(__SIZE_TYPE__);
void free(void *);

struct Test {
  int a;
  int b;
};

static void update_a(int *b) {
  struct Test *head = container_of_typed(b, struct Test, b);
  head->a = 10; // no-warning
}

void scalar_member(void) {
  struct Test object = {0};
  update_a(&object.b);
}

struct Child {
  int value;
};

struct Parent {
  int id;
  struct Child child;
  int tail;
};

static void set_id(struct Child *child) {
  struct Parent *parent =
      container_of_typed(child, struct Parent, child);
  parent->id = 1; // no-warning
}

void direct_member(void) {
  struct Parent object = {0};
  set_id(&object.child);
}

static int read_tail(struct Child *child) {
  struct Parent *parent = container_of(child, struct Parent, child);
  return parent->tail; // no-warning
}

struct Holder {
  struct Parent *parent;
};

int symbolic_parent(struct Holder *holder) {
  return read_tail(&holder->parent->child); // no-warning
}

struct PathList {
  int flags;
};

struct Route {
  char pad[56];
  struct PathList pathlist;
  void *head;
};

struct QueuedRoute {
  struct Route *route;
};

static void bind_pathlist(struct PathList *pathlist) {
  struct Route *route = container_of(pathlist, struct Route, pathlist);
  if (route->head) // no-warning
    (void)0;
}

void symbolic_field_parent(struct QueuedRoute *queued) {
  bind_pathlist(&queued->route->pathlist);
}

struct GrandParent {
  int prefix;
  struct Parent parent;
};

int nested_parent(void) {
  struct GrandParent object = {0};
  struct Parent *parent =
      container_of(&object.parent.child, struct Parent, child);
  return parent->tail; // no-warning
}

void containing_array(void) {
  struct Parent objects[2] = {0};
  struct Parent *parent =
      container_of(&objects[0].child, struct Parent, child);
  parent[1].tail = 1; // no-warning
}

void containing_array_from_second_element(void) {
  struct Parent objects[2] = {0};
  struct Parent *parent =
      container_of(&objects[1].child, struct Parent, child);
  (parent - 1)->id = 1; // no-warning
}

struct FirstMember {
  struct Child child;
  int tail;
};

int zero_offset_field(void) {
  struct FirstMember object = {0};
  struct FirstMember *parent =
      container_of(&object.child, struct FirstMember, child);
  return parent->tail; // no-warning
}

void zero_offset_containing_array(void) {
  struct FirstMember objects[2] = {0};
  struct FirstMember *parent =
      container_of(&objects[0].child, struct FirstMember, child);
  parent[1].tail = 1; // no-warning
}

union ParentUnion {
  struct Child child;
  int value;
};

void union_containing_array(void) {
  union ParentUnion objects[2] = {0};
  union ParentUnion *parent =
      container_of(&objects[0].child, union ParentUnion, child);
  parent[1].value = 1; // no-warning
}

struct PackedParent {
  char tag;
  struct Child child;
  int tail;
} __attribute__((packed));

int packed_parent(void) {
  struct PackedParent object = {0};
  struct PackedParent *parent =
      container_of(&object.child, struct PackedParent, child);
  return parent->tail; // no-warning
}

int unsigned_character_arithmetic(void) {
  struct Parent object = {0};
  struct Parent *parent =
      container_of_uchar(&object.child, struct Parent, child);
  return parent->tail; // no-warning
}

int sufficient_raw_storage(void) {
  unsigned char storage[sizeof(struct Parent)] = {0};
  struct Parent *object = (struct Parent *)storage;
  struct Parent *parent =
      container_of(&object->child, struct Parent, child);
  parent->tail = 1; // no-warning
  return parent->tail; // no-warning
}

int sufficient_heap_storage(void) {
  struct Parent *object = (struct Parent *)malloc(sizeof(*object));
  if (!object)
    return 0;

  struct Parent *parent =
      container_of(&object->child, struct Parent, child);
  parent->tail = 1; // no-warning
  int result = parent->tail; // no-warning
  free(object);
  return result;
}

struct ForwardParent;
struct ForwardParent {
  int id;
  struct Child child;
};

int forward_declared_parent(void) {
  struct ForwardParent object = {0};
  struct ForwardParent *parent =
      container_of(&object.child, struct ForwardParent, child);
  return parent->id; // no-warning
}

int split_adjustment(void) {
  struct Parent object = {0};
  char *address = (char *)&object.child;
  address -= offsetof(struct Parent, child);
  struct Parent *parent = (struct Parent *)address;
  return parent->tail; // no-warning
}

// The matcher relies on region provenance and the ABI field offset, not on an
// OffsetOfExpr surviving in the subtraction expression.
enum { ParentChildOffset = offsetof(struct Parent, child) };

int saved_offset_constant(void) {
  struct Parent object = {0};
  struct Parent *parent =
      (struct Parent *)((char *)&object.child - ParentChildOffset);
  return parent->tail; // no-warning
}

int off_by_one_before_parent(void) {
  struct Parent object = {0};
  struct Parent *parent =
      (struct Parent *)((char *)&object.child -
                        offsetof(struct Parent, child) - 1);
  return parent->id; // expected-warning{{Out of bound access to memory}}
}

struct OtherParent {
  int prefix[2];
  struct Child child;
  int tail;
};

int wrong_parent_type(void) {
  struct Parent object = {0};
  struct OtherParent *parent =
      container_of(&object.child, struct OtherParent, child);
  return parent->prefix[0]; // expected-warning{{Out of bound access to memory}}
}

int raw_storage_with_sufficient_extent(void) {
  unsigned char storage[sizeof(struct Parent)] = {0};
  struct Parent *fake_parent = (struct Parent *)storage;
  struct Parent *parent =
      container_of(&fake_parent->child, struct Parent, child);
  return parent->tail; // no-warning
}

int unrelated_storage(void) {
  int storage = 0;
  struct Parent *fake_parent = (struct Parent *)&storage;
  struct Parent *parent =
      container_of(&fake_parent->child, struct Parent, child);
  return parent->tail; // expected-warning{{Out of bound access to memory}}
}

int insufficient_raw_storage(void) {
  unsigned char storage[sizeof(struct Parent) - 1] = {0};
  struct Parent *fake_parent = (struct Parent *)storage;
  struct Parent *parent =
      container_of(&fake_parent->child, struct Parent, child);
  return parent->tail; // expected-warning{{Out of bound access to memory}}
}

int insufficient_heap_storage(void) {
  struct Parent *object = (struct Parent *)malloc(sizeof(*object) - 1);
  // expected-warning@-1{{allocation of insufficient size}}
  if (!object)
    return 0;

  struct Parent *parent =
      container_of(&object->child, struct Parent, child);
  parent->tail = 1; // expected-warning{{Out of bound access to memory}}
  free(object);
  return 0;
}

int standalone_child(void) {
  struct Child child = {0};
  struct Parent *parent = container_of(&child, struct Parent, child);
  parent->id = 1; // expected-warning{{Out of bound access to memory}}
  return 0;
}

int unrelated_storage_zero_offset(void) {
  int storage = 0;
  struct FirstMember *fake_parent = (struct FirstMember *)&storage;
  struct FirstMember *parent =
      container_of(&fake_parent->child, struct FirstMember, child);
  return parent[1].tail; // expected-warning{{Out of bound access to memory}}
}

struct TwoChildren {
  int id;
  struct Child first;
  struct Child second;
};

int wrong_member_offset(void) {
  struct TwoChildren object = {0};
  struct TwoChildren *parent =
      container_of(&object.first, struct TwoChildren, second);
  return parent->id; // expected-warning{{Out of bound access to memory}}
}

int before_reconstructed_parent(void) {
  struct Parent object = {0};
  struct Parent *parent = container_of(&object.child, struct Parent, child);
  return (parent - 1)->id; // expected-warning{{Out of bound access to memory}}
}

int after_reconstructed_parent(void) {
  struct Parent object = {0};
  struct Parent *parent = container_of(&object.child, struct Parent, child);
  return (parent + 1)->id; // expected-warning{{Out of bound access to memory}}
}

int after_containing_array(void) {
  struct Parent objects[2] = {0};
  struct Parent *parent =
      container_of(&objects[0].child, struct Parent, child);
  return parent[2].id; // expected-warning{{Out of bound access to memory}}
}
