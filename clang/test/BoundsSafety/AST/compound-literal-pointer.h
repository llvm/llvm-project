#pragma clang system_header

// CHECK: |-FunctionDecl {{.+}} compound_from_argument
// CHECK: | `-CompoundLiteralExpr {{.+}} 'int *' lvalue
int *compound_from_argument(int *p) {
    return (int *) { p };
}

// CHECK: |-FunctionDecl {{.+}} compound_from_addrof
// CHECK: | `-CompoundLiteralExpr {{.+}} 'int *' lvalue
int *compound_from_addrof(void) {
    int x;
    return (int *) { &x };
}

// CHECK: |-FunctionDecl {{.+}} compound_from_null
// CHECK: | `-CompoundLiteralExpr {{.+}} 'int *' lvalue
int *compound_from_null(void) {
    int x;
    return (int *) { 0 };
}

// CHECK: |-FunctionDecl {{.+}} compound_from_function_call
// CHECK: | `-CompoundLiteralExpr {{.+}} 'int *' lvalue
int *compound_from_function_call(void) {
    return (int *) { compound_from_null() };
}
