
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

// CHECK: VarDecl {{.+}} g_var_proto 'int *__single(*__single)(int *__single)'
int *(*g_var_proto)(int *p);

// CHECK: VarDecl {{.+}} g_var_proto_va 'int *__single(*__single)(int *__single, ...)'
int *(*g_var_proto_va)(int *p, ...);

// CHECK: VarDecl {{.+}} g_var_no_proto 'int *__single(*__single)()'
int *(*g_var_no_proto)();

void foo(void) {
  // CHECK: VarDecl {{.+}} l_var_proto 'int *__single(*__single)(int *__single)'
  int *(*l_var_proto)(int *p);

  // CHECK: VarDecl {{.+}} l_var_proto_va 'int *__single(*__single)(int *__single, ...)'
  int *(*l_var_proto_va)(int *p, ...);

  // CHECK: VarDecl {{.+}} l_var_no_proto 'int *__single(*__single)()'
  int *(*l_var_no_proto)();
}

// CHECK: ParmVarDecl {{.+}} param_proto 'int *__single(*__single)(int *__single)'
void f1(int *(*param_proto)(int *p));

// CHECK: ParmVarDecl {{.+}} param_proto_va 'int *__single(*__single)(int *__single, ...)'
void f2(int *(*param_proto_va)(int *p, ...));

// CHECK: ParmVarDecl {{.+}} param_no_proto 'int *__single(*__single)()'
void f3(int *(*param_no_proto)());

// CHECK: FunctionDecl {{.+}} ret_proto 'int *__single(*__single(void))(int *__single)'
int *(*ret_proto(void))(int *p);

// CHECK: FunctionDecl {{.+}} ret_proto_va 'int *__single(*__single(void))(int *__single, ...)'
int *(*ret_proto_va(void))(int *p, ...);

// CHECK: FunctionDecl {{.+}} ret_no_proto 'int *__single(*__single(void))()'
int *(*ret_no_proto(void))();

struct bar {
  // CHECK: FieldDecl {{.+}} field_proto 'int *__single(*__single)(int *__single)'
  int *(*field_proto)(int *p);

  // CHECK: FieldDecl {{.+}} field_proto_va 'int *__single(*__single)(int *__single, ...)'
  int *(*field_proto_va)(int *p, ...);

  // CHECK: FieldDecl {{.+}} field_no_proto 'int *__single(*__single)()'
  int *(*field_no_proto)();
};

// CHECK: FunctionDecl {{.+}} nested 'int *__single(*__single(int *__single(*__single)(int *__single)))(int *__single(*__single)(int *__single))'
int *(*nested(int *(*)(int *)))(int *(*)(int *));

// CHECK: VarDecl {{.+}} array 'int *__single(*__single[42])(int *__single)'
int *(*array[42])(int *);

struct digest {
  // CHECK: FieldDecl {{.+}} di 'const struct ccdigest_info *__single(*__single)(void)'
  const struct ccdigest_info *(*di)(void);
};
