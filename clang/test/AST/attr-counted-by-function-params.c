// RUN: %clang_cc1 -fexperimental-late-parse-attributes %s -ast-dump | FileCheck %s

#define __counted_by(f)  __attribute__((counted_by(f)))
#define __counted_by_or_null(f)  __attribute__((counted_by_or_null(f)))
#define __sized_by(f)  __attribute__((sized_by(f)))

// The count parameter is declared first. The attribute is still late-parsed;
// that is decided by the attribute and the language option, not by whether the
// name is already in scope. This pins that both orderings end up with the same
// CountAttributedType.
void back_ref(int count, int *__counted_by(count) buf);
// CHECK-LABEL: FunctionDecl {{.*}} back_ref 'void (int, int * __counted_by(count))'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used count 'int'
// CHECK-NEXT: | `-ParmVarDecl {{.*}} buf 'int * __counted_by(count)':'int *'

// The count parameter is declared after the annotated pointer, so the
// attribute argument can only be parsed once the whole prototype is known.
void fwd_ref(int *__counted_by(count) buf, int count);
// CHECK-LABEL: FunctionDecl {{.*}} fwd_ref 'void (int * __counted_by(count), int)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} buf 'int * __counted_by(count)':'int *'
// CHECK-NEXT: | `-ParmVarDecl {{.*}} used count 'int'

void fwd_ref_or_null(int *__counted_by_or_null(count) buf, int count);
// CHECK-LABEL: FunctionDecl {{.*}} fwd_ref_or_null 'void (int * __counted_by_or_null(count), int)'

void fwd_ref_sized(void *__sized_by(count) buf, int count);
// CHECK-LABEL: FunctionDecl {{.*}} fwd_ref_sized 'void (void * __sized_by(count), int)'

// Two parameters referring to the same count.
void two_buffers(int *__counted_by(count) a, int *__counted_by(count) b,
                 int count);
// CHECK-LABEL: FunctionDecl {{.*}} two_buffers 'void (int * __counted_by(count), int * __counted_by(count), int)'

// The count expression resolves to the ParmVarDecl of this function, not to
// the global of the same name.
int count;
void shadowed(int *__counted_by(count) buf, int count);
// CHECK-LABEL: FunctionDecl {{.*}} shadowed 'void (int * __counted_by(count), int)'
// CHECK-NEXT: |-ParmVarDecl {{.*}} buf 'int * __counted_by(count)':'int *'
// CHECK-NEXT: `-ParmVarDecl {{.*}} used count 'int'
