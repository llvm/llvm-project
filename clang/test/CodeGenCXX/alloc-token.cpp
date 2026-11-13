// RUN: %clang_cc1 -fsanitize=alloc-token -triple x86_64-linux-gnu -std=c++20 -fexceptions -fcxx-exceptions -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

#include "../Analysis/Inputs/system-header-simulator-cxx.h"
extern "C" {
void *aligned_alloc(size_t alignment, size_t size) __attribute__((malloc));
void *malloc(size_t size) __attribute__((malloc));
void free(void *);
void *calloc(size_t num, size_t size) __attribute__((malloc));
void *realloc(void *ptr, size_t size) __attribute__((malloc));
void *reallocarray(void *ptr, size_t nmemb, size_t size) __attribute__((malloc));
void *memalign(size_t alignment, size_t size) __attribute__((malloc));
void *valloc(size_t size) __attribute__((malloc));
void *pvalloc(size_t size) __attribute__((malloc));
int posix_memalign(void **memptr, size_t alignment, size_t size);

struct __sized_ptr_t {
  void *p;
  size_t n;
};
enum class __hot_cold_t : uint8_t;
__sized_ptr_t __size_returning_new(size_t size) __attribute__((malloc_span));
__sized_ptr_t __size_returning_new_hot_cold(size_t, __hot_cold_t) __attribute__((malloc_span));
__sized_ptr_t __size_returning_new_aligned(size_t, std::align_val_t) __attribute__((malloc_span));
__sized_ptr_t __size_returning_new_aligned_hot_cold(size_t, std::align_val_t,  __hot_cold_t) __attribute__((malloc_span));
}

void *sink; // prevent optimizations from removing the calls

// CHECK-LABEL: define dso_local void @_Z16test_malloc_likev(
// CHECK: call noalias ptr @malloc(i64 noundef 4){{.*}} !alloc_token [[META_INT:![0-9]+]]
// CHECK: call noalias ptr @calloc(i64 noundef 3, i64 noundef 4){{.*}} !alloc_token [[META_INT]]
// CHECK: call noalias ptr @realloc(ptr noundef {{.*}}, i64 noundef 8){{.*}} !alloc_token [[META_LONG:![0-9]+]]
// CHECK: call noalias ptr @reallocarray(ptr noundef {{.*}}, i64 noundef 5, i64 noundef 8), !alloc_token [[META_LONG]]
// CHECK: call noalias align 128 ptr @aligned_alloc(i64 noundef 128, i64 noundef 4){{.*}} !alloc_token [[META_INT]]
// CHECK: call noalias ptr @memalign(i64 noundef 16, i64 noundef 4), !alloc_token [[META_INT]]
// CHECK: call noalias ptr @valloc(i64 noundef 4), !alloc_token [[META_INT]]
// CHECK: call noalias ptr @pvalloc(i64 noundef 4), !alloc_token [[META_INT]]
// CHECK: call i32 @posix_memalign(ptr noundef @sink, i64 noundef 64, i64 noundef 4)
void test_malloc_like() {
  sink = malloc(sizeof(int));
  sink = calloc(3, sizeof(int));
  sink = realloc(sink, sizeof(long));
  sink = reallocarray(sink, 5, sizeof(long));
  sink = aligned_alloc(128, sizeof(int));
  sink = memalign(16, sizeof(int));
  sink = valloc(sizeof(int));
  sink = pvalloc(sizeof(int));
  posix_memalign(&sink, 64, sizeof(int)); // FIXME: support posix_memalign
}

class ForwardDecl;

// CHECK-LABEL: define dso_local void @_Z21test_malloc_like_castv(
// CHECK: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_INT]]
// CHECK: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_INT]]
// CHECK-NOT: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_INT]]
void test_malloc_like_cast() {
  sink = (int *)malloc(64);
  sink = reinterpret_cast<int *>(malloc(64));
  // Always fails to assign token ID for incomplete types.
  sink = reinterpret_cast<ForwardDecl *>(malloc(64));
}

// CHECK-LABEL: define dso_local void @_Z17test_operator_newv(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 4){{.*}} !alloc_token [[META_INT]]
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 4){{.*}} !alloc_token [[META_INT]]
void test_operator_new() {
  sink = __builtin_operator_new(sizeof(int));
  sink = ::operator new(sizeof(int));
}

// CHECK-LABEL: define dso_local void @_Z25test_operator_new_nothrowv(
// CHECK: call noalias noundef ptr @_ZnwmRKSt9nothrow_t(i64 noundef 4, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow){{.*}} !alloc_token [[META_INT]]
// CHECK: call noalias noundef ptr @_ZnwmRKSt9nothrow_t(i64 noundef 4, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow){{.*}} !alloc_token [[META_INT]]
void test_operator_new_nothrow() {
  sink = __builtin_operator_new(sizeof(int), std::nothrow);
  sink = ::operator new(sizeof(int), std::nothrow);
}

// CHECK-LABEL: define dso_local noundef ptr @_Z8test_newv(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 4){{.*}} !alloc_token [[META_INT]]
int *test_new() {
  return new int;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z14test_new_arrayv(
// CHECK: call noalias noundef nonnull ptr @_Znam(i64 noundef 40){{.*}} !alloc_token [[META_INT]]
int *test_new_array() {
  return new int[10];
}

// CHECK-LABEL: define dso_local noundef ptr @_Z16test_new_nothrowv(
// CHECK: call noalias noundef ptr @_ZnwmRKSt9nothrow_t(i64 noundef 4, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow){{.*}} !alloc_token [[META_INT]]
int *test_new_nothrow() {
  return new (std::nothrow) int;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z22test_new_array_nothrowv(
// CHECK: call noalias noundef ptr @_ZnamRKSt9nothrow_t(i64 noundef 40, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow){{.*}} !alloc_token [[META_INT]]
int *test_new_array_nothrow() {
  return new (std::nothrow) int[10];
}

// CHECK-LABEL: define dso_local void @_Z23test_size_returning_newv(
// CHECK: call { ptr, i64 } @__size_returning_new(i64 noundef 8){{.*}} !alloc_token [[META_LONG]]
// CHECK: call { ptr, i64 } @__size_returning_new_hot_cold(i64 noundef 8, i8 noundef zeroext 1){{.*}} !alloc_token [[META_LONG]]
// CHECK: call { ptr, i64 } @__size_returning_new_aligned(i64 noundef 8, i64 noundef 32){{.*}} !alloc_token [[META_LONG]]
// CHECK: call { ptr, i64 } @__size_returning_new_aligned_hot_cold(i64 noundef 8, i64 noundef 32, i8 noundef zeroext 1){{.*}}_token [[META_LONG]]
void test_size_returning_new() {
  sink = __size_returning_new(sizeof(long)).p;
  sink = __size_returning_new_hot_cold(sizeof(long), __hot_cold_t{1}).p;
  sink = __size_returning_new_aligned(sizeof(long), std::align_val_t{32}).p;
  sink = __size_returning_new_aligned_hot_cold(sizeof(long), std::align_val_t{32}, __hot_cold_t{1}).p;
}

class TestClass {
public:
  virtual void Foo();
  virtual ~TestClass();
  int data[16];
};

void may_throw();

// CHECK-LABEL: define dso_local noundef ptr @_Z27test_exception_handling_newv(
// CHECK: invoke noalias noundef nonnull ptr @_Znwm(i64 noundef 72)
// CHECK-NEXT: !alloc_token [[META_TESTCLASS:![0-9]+]]
TestClass *test_exception_handling_new() {
  try {
    TestClass *obj = new TestClass();
    may_throw();
    return obj;
  } catch (...) {
    return nullptr;
  }
}

// CHECK-LABEL: define dso_local noundef ptr @_Z14test_new_classv(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 72){{.*}} !alloc_token [[META_TESTCLASS]]
TestClass *test_new_class() {
  TestClass *obj = new TestClass();
  obj->data[0] = 42;
  return obj;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z20test_new_class_arrayv(
// CHECK: call noalias noundef nonnull ptr @_Znam(i64 noundef 728){{.*}} !alloc_token [[META_TESTCLASS]]
TestClass *test_new_class_array() {
  TestClass* arr = new TestClass[10];
  arr[0].data[0] = 123;
  return arr;
}

struct PlacedNoPtr {
  long len;
  unsigned char tag;
  char data[1];
};

struct PlacedWithPtr {
  void *next;
  long len;
};

// Allocation calls whose result is only named by a later placement new. The
// inference order is sizeof, then placement new, then cast.

// CHECK-LABEL: define dso_local noundef ptr @_Z25test_placement_new_directv(
// CHECK: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_PLACEDNOPTR:![0-9]+]]
PlacedNoPtr *test_placement_new_direct() {
  return new (malloc(64)) PlacedNoPtr();
}

// CHECK-LABEL: define dso_local noundef ptr @_Z24test_placement_new_localv(
// CHECK: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_PLACEDWITHPTR:![0-9]+]]
PlacedWithPtr *test_placement_new_local() {
  void *mem = malloc(64);
  if (!mem)
    return nullptr;
  return new (mem) PlacedWithPtr();
}

// CHECK-LABEL: define dso_local noundef ptr @_Z30test_placement_new_sized_localv(
// CHECK: call { ptr, i64 } @__size_returning_new(i64 noundef 64){{.*}} !alloc_token [[META_PLACEDNOPTR]]
PlacedNoPtr *test_placement_new_sized_local() {
  __sized_ptr_t res = __size_returning_new(64);
  PlacedNoPtr *rep = new (res.p) PlacedNoPtr();
  rep->len = res.n;
  return rep;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z29test_placement_new_sized_castv(
// CHECK: call { ptr, i64 } @__size_returning_new_hot_cold(i64 noundef 64, i8 noundef zeroext 1){{.*}} !alloc_token [[META_PLACEDNOPTR]]
PlacedNoPtr *test_placement_new_sized_cast() {
  auto res = __size_returning_new_hot_cold(64, __hot_cold_t{1});
  return new (static_cast<char *>(res.p)) PlacedNoPtr();
}

// CHECK-LABEL: define dso_local noundef ptr @_Z24test_placement_new_arrayv(
// CHECK: call noalias ptr @malloc(i64 noundef 640){{.*}} !alloc_token [[META_PLACEDNOPTR]]
PlacedNoPtr *test_placement_new_array() {
  void *mem = malloc(640);
  return new (mem) PlacedNoPtr[10];
}

// A sizeof in the size argument states what the allocation was sized for and
// takes precedence: the placement new may construct only a header, with the
// elements placed by pointer arithmetic.
// CHECK-LABEL: define dso_local noundef ptr @_Z34test_placement_new_sizeof_prevailsv(
// CHECK: call noalias ptr @malloc(i64 noundef 40){{.*}} !alloc_token [[META_PLACEDWITHPTR]]
// CHECK: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_CHAR:![0-9]+]]
long *test_placement_new_sizeof_prevails() {
  void *mem = malloc(2 * sizeof(PlacedWithPtr) + sizeof(long));
  long *count = new (mem) long(2);
  // The same holds for a sizeof of a character type.
  void *bytes = malloc(sizeof(char) * 64);
  sink = new (bytes) PlacedWithPtr();
  return count;
}

// A cast of the result is weaker evidence than the placement new, whether the
// cast is to a raw storage pointer (char *, void *) or to a base class.
struct DerivedWithPtr : PlacedNoPtr {
  void *next;
};
// CHECK-LABEL: define dso_local void @_Z28test_placement_new_over_castv(
// CHECK: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_PLACEDWITHPTR]]
// CHECK: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_PLACEDWITHPTR]]
// CHECK: call { ptr, i64 } @__size_returning_new(i64 noundef 64){{.*}} !alloc_token [[META_PLACEDWITHPTR]]
// CHECK: call noalias ptr @malloc(i64 noundef 32){{.*}} !alloc_token [[META_DERIVEDWITHPTR:![0-9]+]]
void test_placement_new_over_cast() {
  char *p1 = static_cast<char *>(malloc(64));
  sink = new (p1) PlacedWithPtr();
  auto *p2 = (unsigned char *)malloc(64);
  sink = new (p2) PlacedWithPtr();
  // The member access is an implicit lvalue-to-rvalue cast to void *.
  sink = new (__size_returning_new(64).p) PlacedWithPtr();
  auto *base = static_cast<PlacedNoPtr *>(malloc(32));
  sink = new (base) DerivedWithPtr();
}

// Storage constructed into with different types: no type is inferred.
// CHECK-LABEL: define dso_local void @_Z28test_placement_new_ambiguousi(
// CHECK: call noalias ptr @malloc(i64 noundef 64)
// CHECK-NOT: !alloc_token
// CHECK: ret void
void test_placement_new_ambiguous(int which) {
  void *mem = malloc(64);
  if (which)
    sink = new (mem) PlacedNoPtr();
  else
    sink = new (mem) PlacedWithPtr();
}

// A user-provided placement operator new may use its argument as allocator
// state rather than as the storage, so the allocation feeding it stays untyped
// (the custom operator new call itself is typed).
struct Arena {};
void *operator new(size_t, Arena *);
Arena *alloc_arena(size_t) __attribute__((malloc));
// CHECK-LABEL: define dso_local void @_Z24test_placement_new_arenav(
// CHECK: call noalias noundef ptr @_Z11alloc_arenam(i64 noundef 64)
// CHECK-NOT: @_Z11alloc_arenam{{.*}}!alloc_token
// CHECK: call noundef ptr @_ZnwmP5Arena(i64 noundef 16, {{.*}} !alloc_token [[META_PLACEDNOPTR]]
void test_placement_new_arena() {
  Arena *arena = alloc_arena(64);
  sink = new (arena) PlacedNoPtr();
}

// Same for a class-specific operator new(size_t, void *).
struct ClassPlacement {
  void *next;
  static void *operator new(size_t, void *ctx);
};
// CHECK-LABEL: define dso_local void @_Z33test_placement_new_class_specificv(
// CHECK: call noalias ptr @malloc(i64 noundef 64)
// CHECK-NOT: @malloc{{.*}}!alloc_token
// CHECK: call noundef ptr @_ZN14ClassPlacementnwEmPv(i64 noundef 8, {{.*}} !alloc_token [[META_CLASSPLACEMENT:![0-9]+]]
void test_placement_new_class_specific() {
  void *ctx = malloc(64);
  sink = new (ctx) ClassPlacement;
}

// Indirect member accesses, address-of, and static members name storage other
// than the allocation itself.
struct Holder {
  void *p;
};
Holder *alloc_holder(size_t) __attribute__((malloc));
struct SpanWithStatic {
  void *p;
  size_t n;
  static void *storage;
};
SpanWithStatic alloc_span_with_static(size_t) __attribute__((malloc_span));
// CHECK-LABEL: define dso_local void @_Z27test_placement_new_indirectv(
// CHECK: call noalias noundef ptr @_Z12alloc_holderm(i64 noundef 64)
// CHECK-NOT: !alloc_token
// CHECK: call noalias noundef ptr @_Z12alloc_holderm(i64 noundef 64)
// CHECK-NOT: !alloc_token
// CHECK: call noalias ptr @malloc(i64 noundef 64)
// CHECK-NOT: !alloc_token
// CHECK: call { ptr, i64 } @_Z22alloc_span_with_staticm(i64 noundef 64)
// CHECK-NOT: !alloc_token
// CHECK: ret void
void test_placement_new_indirect() {
  Holder *h = alloc_holder(64);
  sink = new (h->p) PlacedNoPtr();
  Holder *h2 = alloc_holder(64);
  sink = new (&h2->p) PlacedNoPtr();
  void *mem = malloc(64);
  sink = new (&mem) unsigned char(0);
  SpanWithStatic span = alloc_span_with_static(64);
  sink = new (span.storage) PlacedNoPtr();
}

// A variable is used only if the function defines it exactly once, so that
// its value does not depend on control flow. A null pointer initializer does
// not count as a definition.
struct Storage {
  void *p;
  size_t n;
  size_t size() const;
  void reset();
};
Storage alloc_storage(size_t) __attribute__((malloc_span));
void peek(void *const &p);
// CHECK-LABEL: define dso_local void @_Z36test_placement_new_single_definitionb(
// CHECK: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_PLACEDNOPTR]]
// CHECK: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_PLACEDWITHPTR]]
// CHECK: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_PLACEDNOPTR]]
// CHECK: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_PLACEDWITHPTR]]
// CHECK: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_PLACEDNOPTR]]
// CHECK: call { ptr, i64 } @_Z13alloc_storagem(i64 noundef 64){{.*}} !alloc_token [[META_PLACEDWITHPTR]]
void test_placement_new_single_definition(bool c) {
  // Declared without initializer, assigned once.
  void *p;
  p = malloc(64);
  sink = new (p) PlacedNoPtr();
  // Initialized to null, assigned once in a branch.
  void *q = nullptr;
  if (c)
    q = malloc(64);
  if (q)
    sink = new (q) PlacedWithPtr();
  // Defined once, used inside a branch and a loop.
  void *r = malloc(64);
  if (c)
    sink = new (r) PlacedNoPtr();
  void *s = malloc(64);
  for (int i = 0; i < 2; ++i)
    sink = new (s) PlacedWithPtr();
  // Reads through a const reference or a const member function are not
  // writes.
  void *t = malloc(64);
  peek(t);
  sink = new (t) PlacedNoPtr();
  Storage u = alloc_storage(64);
  (void)u.size();
  sink = new (u.p) PlacedWithPtr();
}

// Reassigned variables are not used: the storage the placement new ends up in
// depends on control flow, and the sizeof or cast of the other definition
// would be overridden. Nothing is inferred for any of these allocations.
// CHECK-LABEL: define dso_local void @_Z29test_placement_new_reassignedb(
// CHECK: call noalias ptr @malloc(i64 noundef 8)
// CHECK-NOT: !alloc_token
// CHECK: call noalias ptr @malloc(i64 noundef 8)
// CHECK-NOT: !alloc_token
// CHECK: call noalias ptr @malloc(i64 noundef 64)
// CHECK-NOT: !alloc_token
// CHECK: call noalias ptr @malloc(i64 noundef 32)
// CHECK-NOT: !alloc_token
// CHECK: call noalias ptr @malloc(i64 noundef 64)
// CHECK-NOT: !alloc_token
// CHECK: call noalias ptr @malloc(i64 noundef 32)
// CHECK-NOT: !alloc_token
// CHECK: call noalias ptr @malloc(i64 noundef 64)
// CHECK-NOT: !alloc_token
// CHECK: ret void
void test_placement_new_reassigned(bool c) {
  // Freed and reassigned before the placement new.
  void *p = malloc(8);
  free(p);
  p = malloc(8);
  sink = new (p) long(10);
  // Redefined in one branch of an if, used in the other.
  void *q = malloc(64);
  if (c)
    q = malloc(32);
  else
    sink = new (q) PlacedNoPtr();
  // Captured by reference and reassigned before the lambda runs.
  void *r = malloc(64);
  auto construct = [&] { sink = new (r) PlacedNoPtr(); };
  r = malloc(32);
  construct();
  // The assignment of the placement new's result is a second definition.
  void *s = malloc(64);
  s = new (s) PlacedNoPtr();
}

// A parameter's incoming value is a definition that cannot be seen, so
// parameters are not used.
// CHECK-LABEL: define dso_local void @_Z28test_placement_new_parameterPvb(
// CHECK: call noalias ptr @malloc(i64 noundef 64)
// CHECK-NOT: !alloc_token
// CHECK: ret void
void test_placement_new_parameter(void *p, bool c) {
  if (c)
    p = malloc(64);
  sink = new (p) PlacedNoPtr();
}

// Other writes to the variable disqualify it as well: address-of, compound
// assignment, increment, a write to one of its fields, a non-const member
// call, and binding to a non-const reference, directly, as a call argument,
// or as a constructor argument.
void rebind(void *&p);
struct Rebinder {
  Rebinder(void *&p);
};
// CHECK-LABEL: define dso_local void @_Z31test_placement_new_other_writesv(
// CHECK: call noalias ptr @malloc(i64 noundef 64)
// CHECK-NOT: !alloc_token
// CHECK: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_CHAR]]
// CHECK: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_CHAR]]
// CHECK: call { ptr, i64 } @_Z13alloc_storagem(i64 noundef 64)
// CHECK-NOT: !alloc_token
// CHECK: call { ptr, i64 } @_Z13alloc_storagem(i64 noundef 64)
// CHECK-NOT: !alloc_token
// CHECK: call noalias ptr @malloc(i64 noundef 64)
// CHECK-NOT: !alloc_token
// CHECK: call noalias ptr @malloc(i64 noundef 64)
// CHECK-NOT: !alloc_token
// CHECK: call noalias ptr @malloc(i64 noundef 64)
// CHECK-NOT: !alloc_token
// CHECK: ret void
void test_placement_new_other_writes() {
  void *p = malloc(64);
  sink = &p;
  sink = new (p) PlacedNoPtr();
  // With the variable disqualified, the cast of the result applies instead.
  char *p2 = static_cast<char *>(malloc(64));
  p2 += 8;
  sink = new (p2) PlacedNoPtr();
  char *p3 = static_cast<char *>(malloc(64));
  ++p3;
  sink = new (p3) PlacedNoPtr();
  Storage s1 = alloc_storage(64);
  s1.p = sink;
  sink = new (s1.p) PlacedNoPtr();
  Storage s2 = alloc_storage(64);
  s2.reset();
  sink = new (s2.p) PlacedNoPtr();
  void *q = malloc(64);
  void *&qref = q;
  qref = sink;
  sink = new (q) PlacedNoPtr();
  void *r = malloc(64);
  rebind(r);
  sink = new (r) PlacedNoPtr();
  void *t = malloc(64);
  Rebinder rebinder(t);
  sink = new (t) PlacedNoPtr();
}

// Variables without local storage may be written by any call in between.
void *global_storage;
// CHECK-LABEL: define dso_local void @_Z28test_placement_new_non_localv(
// CHECK: call noalias ptr @malloc(i64 noundef 64)
// CHECK-NOT: !alloc_token
// CHECK: call noalias ptr @malloc(i64 noundef 64)
// CHECK-NOT: !alloc_token
// CHECK: ret void
void test_placement_new_non_local() {
  global_storage = malloc(64);
  sink = new (global_storage) PlacedNoPtr();
  static void *static_storage;
  static_storage = malloc(64);
  sink = new (static_storage) PlacedNoPtr();
}

// Non-generic lambdas use the captured variable; a generic lambda's dependent
// new-expression has no resolved operator new and is skipped.
// CHECK-LABEL: define dso_local void @_Z25test_placement_new_lambdav(
// CHECK: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_PLACEDNOPTR]]
// CHECK: call noalias ptr @malloc(i64 noundef 64)
// CHECK-NOT: @malloc{{.*}}!alloc_token
void test_placement_new_lambda() {
  void *mem = malloc(64);
  auto construct = [mem] { sink = new (mem) PlacedNoPtr(); };
  construct();
  void *mem2 = malloc(64);
  auto construct2 = [mem2](auto x) { sink = new (mem2) decltype(x)(); };
  construct2(PlacedNoPtr{});
}

template <typename T>
T *placement_construct(size_t size) {
  __sized_ptr_t res = __size_returning_new(size);
  return new (res.p) T();
}

// CHECK-LABEL: define dso_local void @_Z27test_placement_new_templatev(
// CHECK: call noundef ptr @_Z19placement_constructI11PlacedNoPtrEPT_m(
// CHECK: call noundef ptr @_Z19placement_constructI13PlacedWithPtrEPT_m(
void test_placement_new_template() {
  sink = placement_construct<PlacedNoPtr>(64);
  sink = placement_construct<PlacedWithPtr>(64);
}

// CHECK-LABEL: define linkonce_odr noundef ptr @_Z19placement_constructI11PlacedNoPtrEPT_m(
// CHECK: call { ptr, i64 } @__size_returning_new(i64 noundef {{.*}}){{.*}} !alloc_token [[META_PLACEDNOPTR]]
// CHECK-LABEL: define linkonce_odr noundef ptr @_Z19placement_constructI13PlacedWithPtrEPT_m(
// CHECK: call { ptr, i64 } @__size_returning_new(i64 noundef {{.*}}){{.*}} !alloc_token [[META_PLACEDWITHPTR]]

// CHECK: [[META_INT]] = !{!"int", i1 false}
// CHECK: [[META_LONG]] = !{!"long", i1 false}
// CHECK: [[META_TESTCLASS]] = !{!"TestClass", i1 true}
// CHECK: [[META_PLACEDNOPTR]] = !{!"PlacedNoPtr", i1 false}
// CHECK: [[META_PLACEDWITHPTR]] = !{!"PlacedWithPtr", i1 true}
// CHECK: [[META_CHAR]] = !{!"char", i1 false}
// CHECK: [[META_DERIVEDWITHPTR]] = !{!"DerivedWithPtr", i1 true}
// CHECK: [[META_CLASSPLACEMENT]] = !{!"ClassPlacement", i1 true}
