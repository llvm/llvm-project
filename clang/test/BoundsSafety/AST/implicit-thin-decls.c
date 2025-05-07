// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s

int* foo(int *);
int* bar(int *parm);
int* baz(int *param) {}

struct S {
    int *i;
    long *l;
};

typedef struct {
    long *l;
    void *v;
} T;

// CHECK: |-FunctionDecl {{.+}} foo 'int *__single(int *__single)'
// CHECK: | `-ParmVarDecl {{.+}} 'int *__single'
// CHECK: |-FunctionDecl {{.+}} bar 'int *__single(int *__single)'
// CHECK: | `-ParmVarDecl {{.+}} parm 'int *__single'
// CHECK: |-FunctionDecl {{.+}} baz 'int *__single(int *__single)'
// CHECK: | |-ParmVarDecl {{.+}} param 'int *__single'
// CHECK: | `-CompoundStmt
// CHECK: |-RecordDecl {{.+}} struct S definition
// CHECK: | |-FieldDecl {{.+}} i 'int *__single'
// CHECK: | `-FieldDecl {{.+}} l 'long *__single'
// CHECK: |-RecordDecl [[ADDR:0x[a-z0-9]+]] {{.+}} struct definition
// CHECK: | |-FieldDecl {{.+}} l 'long *__single'
// CHECK: | `-FieldDecl {{.+}} v 'void *__single'
// CHECK: `-TypedefDecl {{.+}} T 'struct T':'T'
// CHECK:   `-ElaboratedType {{.+}} 'struct T' sugar
// CHECK:     `-RecordType {{.+}} 'T'
// CHECK:       `-Record [[ADDR]] {{.+}}
