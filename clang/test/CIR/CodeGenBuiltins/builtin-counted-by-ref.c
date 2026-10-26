// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// The flexible array member argument used to be emitted as a scalar by the
// target-specific builtin fallback, which crashed (issue #231059). Check other
// targets too, since each of them took a different fallback path.
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple riscv64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s -check-prefix=CIR

struct S {
  char c;
  int arr[] __attribute__((counted_by(c)));
};

void test_issue_231059(struct S *s, int size) {
  *__builtin_counted_by_ref(s->arr) = size;
}

// CIR-LABEL: cir.func {{.*}} @test_issue_231059(
// CIR:   %[[S_ADDR:.*]] = cir.alloca "s"
// CIR:   %[[SIZE_ADDR:.*]] = cir.alloca "size"
// CIR:   %[[SIZE:.*]] = cir.load {{.*}} %[[SIZE_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[SIZE_TRUNC:.*]] = cir.cast integral %[[SIZE]] : !s32i -> !s8i
// CIR:   %[[S:.*]] = cir.load {{.*}} %[[S_ADDR]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:   %[[COUNT_PTR:.*]] = cir.get_member %[[S]][0] {name = "c"} : !cir.ptr<!rec_S> -> !cir.ptr<!s8i>
// CIR:   cir.store {{.*}} %[[SIZE_TRUNC]], %[[COUNT_PTR]] : !s8i, !cir.ptr<!s8i>

// LLVM-LABEL: define {{.*}} void @test_issue_231059(
// LLVM:   %[[SIZE:.*]] = load i32, ptr %{{.*}}
// LLVM:   %[[SIZE_TRUNC:.*]] = trunc i32 %[[SIZE]] to i8
// LLVM:   %[[S:.*]] = load ptr, ptr %{{.*}}
// LLVM:   %[[COUNT_PTR:.*]] = getelementptr inbounds nuw %struct.S, ptr %[[S]], i32 0, i32 0
// LLVM:   store i8 %[[SIZE_TRUNC]], ptr %[[COUNT_PTR]], align 1

// OGCG-LABEL: define {{.*}} void @test_issue_231059(
// OGCG:   %[[SIZE:.*]] = load i32, ptr %{{.*}}
// OGCG:   %[[SIZE_TRUNC:.*]] = trunc i32 %[[SIZE]] to i8
// OGCG:   %[[S:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[COUNT_PTR:.*]] = getelementptr inbounds %struct.S, ptr %[[S]], i32 0, i32 0
// OGCG:   store i8 %[[SIZE_TRUNC]], ptr %[[COUNT_PTR]], align 1

struct a {
  char x;
  short count;
  int array[] __attribute__((counted_by(count)));
};

void test_basic(struct a *p, int size) {
  *__builtin_counted_by_ref(p->array) = size;
}

// CIR-LABEL: cir.func {{.*}} @test_basic(
// CIR:   %[[SIZE_TRUNC:.*]] = cir.cast integral %{{.*}} : !s32i -> !s16i
// CIR:   %[[P:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!rec_a>>, !cir.ptr<!rec_a>
// CIR:   %[[COUNT_PTR:.*]] = cir.get_member %[[P]][1] {name = "count"} : !cir.ptr<!rec_a> -> !cir.ptr<!s16i>
// CIR:   cir.store {{.*}} %[[SIZE_TRUNC]], %[[COUNT_PTR]] : !s16i, !cir.ptr<!s16i>

// LLVM-LABEL: define {{.*}} void @test_basic(
// LLVM:   %[[SIZE_TRUNC:.*]] = trunc i32 %{{.*}} to i16
// LLVM:   %[[P:.*]] = load ptr, ptr %{{.*}}
// LLVM:   %[[COUNT_PTR:.*]] = getelementptr inbounds nuw %struct.a, ptr %[[P]], i32 0, i32 1
// LLVM:   store i16 %[[SIZE_TRUNC]], ptr %[[COUNT_PTR]], align 2

// OGCG-LABEL: define {{.*}} void @test_basic(
// OGCG:   %[[SIZE_TRUNC:.*]] = trunc i32 %{{.*}} to i16
// OGCG:   %[[P:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[COUNT_PTR:.*]] = getelementptr inbounds %struct.a, ptr %[[P]], i32 0, i32 1
// OGCG:   store i16 %[[SIZE_TRUNC]], ptr %[[COUNT_PTR]], align 2

// The count field is reached through several anonymous structs.
struct b {
  int _filler;
  struct {
    int __filler;
    struct {
      int ___filler;
      struct {
        char count;
      };
    };
  };
  struct {
    int filler_;
    struct {
      int filler__;
      struct {
        long array[] __attribute__((counted_by(count)));
      };
    };
  };
};

void test_nested_anon(struct b *p, int size) {
  *__builtin_counted_by_ref(p->array) = size;
}

// CIR-LABEL: cir.func {{.*}} @test_nested_anon(
// CIR:   %[[SIZE_TRUNC:.*]] = cir.cast integral %{{.*}} : !s32i -> !s8i
// CIR:   %[[P:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!rec_b>>, !cir.ptr<!rec_b>
// CIR:   %[[ANON0:.*]] = cir.get_member %[[P]][1] {name = ""} : !cir.ptr<!rec_b> -> !cir.ptr<![[REC_ANON0:.*]]>
// CIR:   %[[ANON1:.*]] = cir.get_member %[[ANON0]][1] {name = ""} : !cir.ptr<![[REC_ANON0]]> -> !cir.ptr<![[REC_ANON1:.*]]>
// CIR:   %[[ANON2:.*]] = cir.get_member %[[ANON1]][1] {name = ""} : !cir.ptr<![[REC_ANON1]]> -> !cir.ptr<![[REC_ANON2:.*]]>
// CIR:   %[[COUNT_PTR:.*]] = cir.get_member %[[ANON2]][0] {name = "count"} : !cir.ptr<![[REC_ANON2]]> -> !cir.ptr<!s8i>
// CIR:   cir.store {{.*}} %[[SIZE_TRUNC]], %[[COUNT_PTR]] : !s8i, !cir.ptr<!s8i>

// LLVM-LABEL: define {{.*}} void @test_nested_anon(
// LLVM:   %[[SIZE_TRUNC:.*]] = trunc i32 %{{.*}} to i8
// LLVM:   %[[P:.*]] = load ptr, ptr %{{.*}}
// LLVM:   %[[ANON0:.*]] = getelementptr inbounds nuw %struct.b, ptr %[[P]], i32 0, i32 1
// LLVM:   %[[ANON1:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %[[ANON0]], i32 0, i32 1
// LLVM:   %[[ANON2:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %[[ANON1]], i32 0, i32 1
// LLVM:   %[[COUNT_PTR:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %[[ANON2]], i32 0, i32 0
// LLVM:   store i8 %[[SIZE_TRUNC]], ptr %[[COUNT_PTR]], align 1

// OGCG-LABEL: define {{.*}} void @test_nested_anon(
// OGCG:   %[[SIZE_TRUNC:.*]] = trunc i32 %{{.*}} to i8
// OGCG:   %[[P:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[COUNT_PTR:.*]] = getelementptr inbounds %struct.b, ptr %[[P]], i32 0, i32 1, i32 1, i32 1, i32 0
// OGCG:   store i8 %[[SIZE_TRUNC]], ptr %[[COUNT_PTR]], align 1

// Without a counted_by attribute, the builtin returns '(void *) 0'.
struct c {
  char x;
  short count;
  int array[];
};

int test_no_counted_by(struct c *p) {
  if (__builtin_counted_by_ref(p->array))
    return 1;
  return 0;
}

// CIR-LABEL: cir.func {{.*}} @test_no_counted_by(
// CIR:   %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR:   %[[COND:.*]] = cir.cast ptr_to_bool %[[NULL]] : !cir.ptr<!void> -> !cir.bool
// CIR:   cir.if %[[COND]]

// LLVM-LABEL: define {{.*}} i32 @test_no_counted_by(
// LLVM:   br i1 false

// OGCG-LABEL: define {{.*}} i32 @test_no_counted_by(
// OGCG:   br i1 false

// The usual idiom to set the count only when there is one.
void test_generic(struct c *p, int size) {
  unsigned long int __ignored;

  *_Generic(
    __builtin_counted_by_ref(p->array),
      void *: &__ignored,
      default: __builtin_counted_by_ref(p->array)) = size;
}

// CIR-LABEL: cir.func {{.*}} @test_generic(
// CIR:   %[[IGNORED:.*]] = cir.alloca "__ignored"
// CIR:   %[[SIZE_EXT:.*]] = cir.cast integral %{{.*}} : !s32i -> !u64i
// CIR:   cir.store {{.*}} %[[SIZE_EXT]], %[[IGNORED]] : !u64i, !cir.ptr<!u64i>

// LLVM-LABEL: define {{.*}} void @test_generic(
// LLVM:   %[[IGNORED:.*]] = alloca i64
// LLVM:   %[[SIZE_EXT:.*]] = sext i32 %{{.*}} to i64
// LLVM:   store i64 %[[SIZE_EXT]], ptr %[[IGNORED]]

// OGCG-LABEL: define {{.*}} void @test_generic(
// OGCG:   %[[IGNORED:.*]] = alloca i64
// OGCG:   %[[SIZE_EXT:.*]] = sext i32 %{{.*}} to i64
// OGCG:   store i64 %[[SIZE_EXT]], ptr %[[IGNORED]]

// A pointer member with a counted_by attribute.
struct d {
  int x;
  short count;
  int *ptr __attribute__((counted_by(count)));
};

void test_pointer_member(struct d *p, int size) {
  *__builtin_counted_by_ref(p->ptr) = size;
}

// CIR-LABEL: cir.func {{.*}} @test_pointer_member(
// CIR:   %[[SIZE_TRUNC:.*]] = cir.cast integral %{{.*}} : !s32i -> !s16i
// CIR:   %[[P:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!rec_d>>, !cir.ptr<!rec_d>
// CIR:   %[[COUNT_PTR:.*]] = cir.get_member %[[P]][1] {name = "count"} : !cir.ptr<!rec_d> -> !cir.ptr<!s16i>
// CIR:   cir.store {{.*}} %[[SIZE_TRUNC]], %[[COUNT_PTR]] : !s16i, !cir.ptr<!s16i>

// LLVM-LABEL: define {{.*}} void @test_pointer_member(
// LLVM:   %[[P:.*]] = load ptr, ptr %{{.*}}
// LLVM:   %[[COUNT_PTR:.*]] = getelementptr inbounds nuw %struct.d, ptr %[[P]], i32 0, i32 1
// LLVM:   store i16 %{{.*}}, ptr %[[COUNT_PTR]], align 2

// OGCG-LABEL: define {{.*}} void @test_pointer_member(
// OGCG:   %[[P:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[COUNT_PTR:.*]] = getelementptr inbounds %struct.d, ptr %[[P]], i32 0, i32 1
// OGCG:   store i16 %{{.*}}, ptr %[[COUNT_PTR]], align 2

// The flexible array member's struct is named and nested in another struct
// (issue #182575).
struct outer {
  struct inner {
    int counter;
    int ent[] __attribute__((counted_by(counter)));
  } *entries;
};

void test_named_nested(struct inner *entries, int count) {
  *__builtin_counted_by_ref(entries->ent) = count;
}

// CIR-LABEL: cir.func {{.*}} @test_named_nested(
// CIR:   %[[ENTRIES:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!rec_inner>>, !cir.ptr<!rec_inner>
// CIR:   %[[COUNT_PTR:.*]] = cir.get_member %[[ENTRIES]][0] {name = "counter"} : !cir.ptr<!rec_inner> -> !cir.ptr<!s32i>
// CIR:   cir.store {{.*}} %{{.*}}, %[[COUNT_PTR]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}} void @test_named_nested(
// LLVM:   %[[COUNT_PTR:.*]] = getelementptr inbounds nuw %struct.inner, ptr %{{.*}}, i32 0, i32 0
// LLVM:   store i32 %{{.*}}, ptr %[[COUNT_PTR]], align 4

// OGCG-LABEL: define {{.*}} void @test_named_nested(
// OGCG:   %[[COUNT_PTR:.*]] = getelementptr inbounds %struct.inner, ptr %{{.*}}, i32 0, i32 0
// OGCG:   store i32 %{{.*}}, ptr %[[COUNT_PTR]], align 4

// The base is not a pointer, so it is emitted as an lvalue.
void test_dot(int size) {
  struct a s;
  *__builtin_counted_by_ref(s.array) = size;
}

// CIR-LABEL: cir.func {{.*}} @test_dot(
// CIR:   %[[S:.*]] = cir.alloca "s" {{.*}} : !cir.ptr<!rec_a>
// CIR:   %[[SIZE_TRUNC:.*]] = cir.cast integral %{{.*}} : !s32i -> !s16i
// CIR:   %[[COUNT_PTR:.*]] = cir.get_member %[[S]][1] {name = "count"} : !cir.ptr<!rec_a> -> !cir.ptr<!s16i>
// CIR:   cir.store {{.*}} %[[SIZE_TRUNC]], %[[COUNT_PTR]] : !s16i, !cir.ptr<!s16i>

// LLVM-LABEL: define {{.*}} void @test_dot(
// LLVM:   %[[S:.*]] = alloca %struct.a
// LLVM:   %[[COUNT_PTR:.*]] = getelementptr inbounds nuw %struct.a, ptr %[[S]], i32 0, i32 1
// LLVM:   store i16 %{{.*}}, ptr %[[COUNT_PTR]], align 2

// OGCG-LABEL: define {{.*}} void @test_dot(
// OGCG:   %[[S:.*]] = alloca %struct.a
// OGCG:   %[[COUNT_PTR:.*]] = getelementptr inbounds %struct.a, ptr %[[S]], i32 0, i32 1
// OGCG:   store i16 %{{.*}}, ptr %[[COUNT_PTR]], align 2

// The count field is addressed from 'p->next', not from 'p'.
struct n {
  struct n *next;
  int count;
  int array[] __attribute__((counted_by(count)));
};

void test_member_base(struct n *p, int size) {
  *__builtin_counted_by_ref(p->next->array) = size;
}

// CIR-LABEL: cir.func {{.*}} @test_member_base(
// CIR:   %[[P:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!rec_n>>, !cir.ptr<!rec_n>
// CIR:   %[[NEXT_ADDR:.*]] = cir.get_member %[[P]][0] {name = "next"} : !cir.ptr<!rec_n> -> !cir.ptr<!cir.ptr<!rec_n>>
// CIR:   %[[NEXT:.*]] = cir.load {{.*}} %[[NEXT_ADDR]] : !cir.ptr<!cir.ptr<!rec_n>>, !cir.ptr<!rec_n>
// CIR:   %[[COUNT_PTR:.*]] = cir.get_member %[[NEXT]][1] {name = "count"} : !cir.ptr<!rec_n> -> !cir.ptr<!s32i>
// CIR:   cir.store {{.*}} %{{.*}}, %[[COUNT_PTR]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}} void @test_member_base(
// LLVM:   %[[P:.*]] = load ptr, ptr %{{.*}}
// LLVM:   %[[NEXT_ADDR:.*]] = getelementptr inbounds nuw %struct.n, ptr %[[P]], i32 0, i32 0
// LLVM:   %[[NEXT:.*]] = load ptr, ptr %[[NEXT_ADDR]]
// LLVM:   %[[COUNT_PTR:.*]] = getelementptr inbounds nuw %struct.n, ptr %[[NEXT]], i32 0, i32 1
// LLVM:   store i32 %{{.*}}, ptr %[[COUNT_PTR]], align 4

// OGCG-LABEL: define {{.*}} void @test_member_base(
// OGCG:   %[[P:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[NEXT_ADDR:.*]] = getelementptr inbounds nuw %struct.n, ptr %[[P]], i32 0, i32 0
// OGCG:   %[[NEXT:.*]] = load ptr, ptr %[[NEXT_ADDR]]
// OGCG:   %[[COUNT_PTR:.*]] = getelementptr inbounds %struct.n, ptr %[[NEXT]], i32 0, i32 1
// OGCG:   store i32 %{{.*}}, ptr %[[COUNT_PTR]], align 4

// The count field is in a struct inside an anonymous union.
struct u {
  int x;
  union {
    struct {
      long count;
    };
    int i;
  };
  int array[] __attribute__((counted_by(count)));
};

void test_union(struct u *p, int size) {
  *__builtin_counted_by_ref(p->array) = size;
}

// CIR-LABEL: cir.func {{.*}} @test_union(
// CIR:   %[[SIZE_EXT:.*]] = cir.cast integral %{{.*}} : !s32i -> !s64i
// CIR:   %[[P:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!rec_u>>, !cir.ptr<!rec_u>
// CIR:   %[[UNION:.*]] = cir.get_member %[[P]][1] {name = ""} : !cir.ptr<!rec_u> -> !cir.ptr<![[REC_UNION:.*]]>
// CIR:   %[[STRUCT:.*]] = cir.get_member %[[UNION]][0] {name = ""} : !cir.ptr<![[REC_UNION]]> -> !cir.ptr<![[REC_STRUCT:.*]]>
// CIR:   %[[COUNT_PTR:.*]] = cir.get_member %[[STRUCT]][0] {name = "count"} : !cir.ptr<![[REC_STRUCT]]> -> !cir.ptr<!s64i>
// CIR:   cir.store {{.*}} %[[SIZE_EXT]], %[[COUNT_PTR]] : !s64i, !cir.ptr<!s64i>

// LLVM-LABEL: define {{.*}} void @test_union(
// LLVM:   %[[SIZE_EXT:.*]] = sext i32 %{{.*}} to i64
// LLVM:   %[[P:.*]] = load ptr, ptr %{{.*}}
// LLVM:   %[[UNION:.*]] = getelementptr inbounds nuw %struct.u, ptr %[[P]], i32 0, i32 1
// LLVM:   %[[COUNT_PTR:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %[[UNION]], i32 0, i32 0
// LLVM:   store i64 %[[SIZE_EXT]], ptr %[[COUNT_PTR]], align 8

// OGCG-LABEL: define {{.*}} void @test_union(
// OGCG:   %[[SIZE_EXT:.*]] = sext i32 %{{.*}} to i64
// OGCG:   %[[P:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[COUNT_PTR:.*]] = getelementptr inbounds %struct.u, ptr %[[P]], i32 0, i32 1, i32 0, i32 0
// OGCG:   store i64 %[[SIZE_EXT]], ptr %[[COUNT_PTR]], align 8
