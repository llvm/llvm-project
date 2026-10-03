// RUN: not %clang_cc1 -fblocks -fsyntax-only -fdiagnostics-print-source-range-info %s 2>&1 | FileCheck %s

void f(int i) __attribute__((format_arg(1)));
// CHECK: attr-source-range.cpp:3:30:{3:41-3:42}{3:8-3:13}

void g(int i, ...) __attribute__((format(printf, 1, 1)));
// CHECK: attr-source-range.cpp:6:35:{6:50-6:51}{6:8-6:13}

int h(void) __attribute__((returns_nonnull));
// CHECK: attr-source-range.cpp:9:28:{9:1-9:4}

void i(int j) __attribute__((nonnull(1)));
// CHECK: attr-source-range.cpp:12:30:{12:38-12:39}{12:8-12:13}

void j(__attribute__((nonnull)) int i);
// CHECK: attr-source-range.cpp:15:23:{15:8-15:38}

void alloc_align_function_pointer(
    char *(*fn)(char *) __attribute__((alloc_align(1))));
// CHECK: attr-source-range.cpp:[[@LINE-1]]:52:{[[@LINE-1]]:17-[[@LINE-1]]:23}: error: 'alloc_align' attribute argument may only refer to a function parameter of integer type

// A top-level qualifier wraps the PointerTypeLoc in a QualifiedTypeLoc. Make
// sure parameter lookup strips that wrapper before finding FunctionProtoTypeLoc.
void alloc_align_qualified_function_pointer(
    char *(*const fn)(char *) __attribute__((alloc_align(1))));
// CHECK: attr-source-range.cpp:[[@LINE-1]]:58:{[[@LINE-1]]:23-[[@LINE-1]]:29}: error: 'alloc_align' attribute argument may only refer to a function parameter of integer type

struct S;
void alloc_align_member_function_pointer(
    char *(S::*fn)(char *) __attribute__((alloc_align(1))));
// CHECK: attr-source-range.cpp:[[@LINE-1]]:55:{[[@LINE-1]]:20-[[@LINE-1]]:26}: error: 'alloc_align' attribute argument may only refer to a function parameter of integer type

void alloc_align_function_reference(
    char *(&fn)(char *) __attribute__((alloc_align(1))));
// CHECK: attr-source-range.cpp:[[@LINE-1]]:52:{[[@LINE-1]]:17-[[@LINE-1]]:23}: error: 'alloc_align' attribute argument may only refer to a function parameter of integer type

void alloc_align_block_pointer(
    char *(^fn)(char *) __attribute__((alloc_align(1))));
// CHECK: attr-source-range.cpp:[[@LINE-1]]:52:{[[@LINE-1]]:17-[[@LINE-1]]:23}: error: 'alloc_align' attribute argument may only refer to a function parameter of integer type

char *reference_target(char *);
char *(&alloc_align_function_reference_variable)(char *)
    __attribute__((alloc_align(1))) = reference_target;
// CHECK: attr-source-range.cpp:[[@LINE-1]]:32:{[[@LINE-2]]:50-[[@LINE-2]]:56}: error: 'alloc_align' attribute argument may only refer to a function parameter of integer type

struct MemberPointerField {
  char *(S::*fn)(char *) __attribute__((alloc_align(1)));
};
// CHECK: attr-source-range.cpp:[[@LINE-2]]:53:{[[@LINE-2]]:18-[[@LINE-2]]:24}: error: 'alloc_align' attribute argument may only refer to a function parameter of integer type

struct StaticDataMember {
  static char *(*fn)(char *) __attribute__((alloc_align(1)));
};
// CHECK: attr-source-range.cpp:[[@LINE-2]]:57:{[[@LINE-2]]:22-[[@LINE-2]]:28}: error: 'alloc_align' attribute argument may only refer to a function parameter of integer type

// type alias (TypedefNameDecl)
using alloc_align_alias __attribute__((alloc_align(1))) = char *(*)(char *);
// CHECK: attr-source-range.cpp:[[@LINE-1]]:52:{[[@LINE-1]]:69-[[@LINE-1]]:75}: error: 'alloc_align' attribute argument may only refer to a function parameter of integer type

// typedef of a function type
typedef char *alloc_align_function_t(char *) __attribute__((alloc_align(1)));
// CHECK: attr-source-range.cpp:[[@LINE-1]]:73:{[[@LINE-1]]:38-[[@LINE-1]]:44}: error: 'alloc_align' attribute argument may only refer to a function parameter of integer type
