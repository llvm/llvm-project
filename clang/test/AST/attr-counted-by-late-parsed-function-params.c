// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify -Wno-visibility -ast-dump %s | FileCheck %s

#define __counted_by(f)  __attribute__((counted_by(f)))
#define __sized_by(f)  __attribute__((sized_by(f)))

typedef int *int_ptr;

// CHECK: FunctionDecl {{.*}} fwd 'void (int * __counted_by(count), int)'
// CHECK-NEXT: ParmVarDecl {{.*}} buf 'int * __counted_by(count)':'int *'
void fwd(int *__counted_by(count) buf, int count);

// CHECK: FunctionDecl {{.*}} declspec 'void (int_ptr __counted_by(count), int)'
void declspec(int_ptr __counted_by(count) buf, int count);

// An array parameter's count moves to the pointer it adjusts to.
// CHECK: FunctionDecl {{.*}} array 'void (int * __sized_by(count), int)'
// CHECK-NEXT: ParmVarDecl {{.*}} arr 'int * __sized_by(count)':'int *'
void array(int arr[] __sized_by(count), int count);
// CHECK: FunctionDecl {{.*}} array_const 'void (int *const __counted_by(count), int)'
void array_const(int arr[const] __counted_by(count), int count);

// The array's qualifiers apply to its element (C99 6.7.3p8), also when they are
// outside the count, as when it is written in the declaration specifiers.
typedef int int_array[];
// CHECK: FunctionDecl {{.*}} array_qualified 'void (int, const int * __counted_by(count))'
// CHECK-NEXT: ParmVarDecl {{.*}} count 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} arr 'const int * __counted_by(count)':'const int *'
void array_qualified(int count, const int_array __counted_by(count) arr);
// CHECK: FunctionDecl {{.*}} array_qualified_late 'void (volatile int * __counted_by(count), int)'
void array_qualified_late(volatile int_array __counted_by(count) arr, int count);

// As for a field, a rejected count stays in the parameter's type and the
// parameter is invalid, which makes its function invalid too.
// CHECK: FunctionDecl {{.*}} invalid rejected 'void (float, int * __counted_by(count))'
// CHECK-NEXT: ParmVarDecl {{.*}} count 'float'
// CHECK-NEXT: ParmVarDecl {{.*}} invalid buf 'int * __counted_by(count)':'int *'
// expected-error@+1{{'counted_by' requires a non-boolean integer type argument}}
void rejected(float count, int *__counted_by(count) buf);

// One that fails to parse becomes an error expression rather than an empty
// count.
// CHECK: FunctionDecl {{.*}} invalid unparseable 'void (int * __counted_by(<recovery-expr>()), int)'
// expected-error@+1{{expected expression}}
void unparseable(int *__counted_by(+) buf, int count);

// A count on a sized array is rejected; the pointer the array adjusts to has
// neither.
// CHECK: FunctionDecl {{.*}} invalid sized_array 'void (int *, int)'
// expected-error@+1{{'counted_by' cannot be applied to an array parameter with an explicit size}}
void sized_array(int arr[10] __counted_by(count), int count);

// A callback's return count names the callback's parameter.
// CHECK: FunctionDecl {{.*}} cb_return 'void (int * __counted_by(len)(*)(int), int)'
void cb_return(int *__counted_by(len) (*cb)(int len), int len);

// CHECK: FunctionDecl {{.*}} cb_param 'void (void (*)(int * __counted_by(m), int))'
void cb_param(void (*cb)(int *__counted_by(m) p, int m));

// An out parameter's count, here a dereferenced parameter, is on the pointer
// that the parameter points to.
// CHECK: FunctionDecl {{.*}} out 'void (int * __counted_by(*len)*, int *)'
// CHECK-NEXT: ParmVarDecl {{.*}} buf 'int * __counted_by(*len)*'
void out(int *__counted_by(*len) *buf, int *len);

// An array parameter's elements may be counted pointers too, whatever its size.
// CHECK: FunctionDecl {{.*}} out_array 'void (int * __counted_by(count)*, int)'
void out_array(int *__counted_by(count) buf[4], int count);

// If the parameter's own pointer has a count too, the nested one is dropped.
// CHECK: FunctionDecl {{.*}} both_counted 'void (int ** __counted_by(m), int, int)'
// expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
void both_counted(int *__counted_by(n) *__counted_by(m) buf, int n, int m);

// CHECK: FunctionDecl {{.*}} unnamed_first 'void (void (*)(int_ptr __counted_by(n), int))'
void unnamed_first(void (__counted_by(n) int_ptr, int n));

// A rejected return count on a function-typed parameter stays, and the
// parameter is invalid.
// CHECK: FunctionDecl {{.*}} invalid fn_typed 'void (int * __counted_by(len)(*)(int))'
// expected-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'int * __counted_by(len)(int)' (aka 'int *(int)') is a function type}}
void fn_typed(int *__counted_by(len) cb(int len));

// A field's own count names the field; a callback's return count names the
// callback's parameter even when a field has the same name, which then stays
// unreferenced.
// CHECK: RecordDecl {{.*}} struct callbacks definition
// CHECK-NEXT: FieldDecl {{.*}} col:{{[0-9]+}} len 'int'
// CHECK-NEXT: FieldDecl {{.*}} ret 'void * __sized_by(len)(*)(int)'
struct callbacks {
  int len;
  void *__sized_by(len) (*ret)(int len);
};

// CHECK: RecordDecl {{.*}} struct own_count definition
// CHECK-NEXT: FieldDecl {{.*}} referenced len 'int'
// CHECK-NEXT: FieldDecl {{.*}} buf 'int * __counted_by(len)':'int *'
struct own_count {
  int len;
  int *__counted_by(len) buf;
};

// A record defined in a parameter clause completes its own members.
// CHECK: FunctionDecl {{.*}} record_in_params
// CHECK: MemberExpr {{.*}} 'int * __counted_by(n)':'int *' lvalue ->p
void record_in_params(struct in_params { int *__counted_by(n) p; int n; } *s) {
  (void)s->p;
}
