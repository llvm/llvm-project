

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -triple arm64-apple-ios -target-feature +sve 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -triple arm64-apple-ios -target-feature +sve %s 2>&1 | FileCheck %s

typedef struct opaque_s *opaque_t;

// CHECK:      VarDecl {{.+}} g_void 'void *__single' cinit
// CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'void *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:   `-CStyleCastExpr {{.+}} 'void *__bidi_indexable' <BitCast>
// CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'unsigned char *__bidi_indexable' <ArrayToPointerDecay> part_of_explicit_cast
// CHECK-NEXT:       `-CompoundLiteralExpr {{.+}} 'unsigned char[1]' lvalue
// CHECK-NEXT:         `-InitListExpr {{.+}} 'unsigned char[1]
void *g_void = (void *)(unsigned char[1]){};

// CHECK:      VarDecl {{.+}} g_opaque 'struct opaque_s *__single' cinit
// CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'struct opaque_s *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:   `-CStyleCastExpr {{.+}} 'struct opaque_s *__bidi_indexable' <BitCast>
// CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'unsigned char *__bidi_indexable' <ArrayToPointerDecay> part_of_explicit_cast
// CHECK-NEXT:       `-CompoundLiteralExpr {{.+}} 'unsigned char[1]' lvalue
// CHECK-NEXT:         `-InitListExpr {{.+}} 'unsigned char[1]'
opaque_t g_opaque = (opaque_t)(unsigned char[1]){};

// CHECK:      VarDecl {{.+}} g_func 'void (*__single)(int)' cinit
// CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'void (*__single)(int)' <BoundsSafetyPointerCast>
// CHECK-NEXT:   `-CStyleCastExpr {{.+}} 'void (*__bidi_indexable)(int)' <BitCast>
// CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'unsigned char *__bidi_indexable' <ArrayToPointerDecay> part_of_explicit_cast
// CHECK-NEXT:       `-CompoundLiteralExpr {{.+}} 'unsigned char[1]' lvalue
// CHECK-NEXT:         `-InitListExpr {{.+}} 'unsigned char[1]'
void (*g_func)(int) = (void (*)(int))(unsigned char[1]){};

// CHECK:      VarDecl {{.+}} g_sizeless '__SVInt8_t *__single' cinit
// CHECK-NEXT: `-ImplicitCastExpr {{.+}} '__SVInt8_t *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:   `-CStyleCastExpr {{.+}} '__SVInt8_t *__bidi_indexable' <BitCast>
// CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'unsigned char *__bidi_indexable' <ArrayToPointerDecay> part_of_explicit_cast
// CHECK-NEXT:       `-CompoundLiteralExpr {{.+}} 'unsigned char[1]' lvalue
// CHECK-NEXT:         `-InitListExpr {{.+}} 'unsigned char[1]'
__SVInt8_t *g_sizeless = (__SVInt8_t *)(unsigned char[1]){};

int f_void(void *);
int f_opaque(opaque_t);
int f_func(void (*)(int));
int f_sizeless(__SVInt8_t *);

void foo(void) {
  int result;

  // CHECK:      CallExpr {{.+}} 'int'
  // CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int (*__single)(void *__single)' <FunctionToPointerDecay>
  // CHECK-NEXT: | `-DeclRefExpr {{.+}} 'int (void *__single)' Function {{.+}} 'f_void' 'int (void *__single)'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'void *__single' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-CStyleCastExpr {{.+}} 'void *__bidi_indexable' <BitCast>
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'unsigned char *__bidi_indexable' <ArrayToPointerDecay> part_of_explicit_cast
  // CHECK-NEXT:       `-CompoundLiteralExpr {{.+}} 'unsigned char[1]' lvalue
  // CHECK-NEXT:         `-InitListExpr {{.+}} 'unsigned char[1]'
  result = f_void((void *)(unsigned char[1]){});

  // CHECK:      CallExpr {{.+}} 'int'
  // CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int (*__single)(struct opaque_s *__single)' <FunctionToPointerDecay>
  // CHECK-NEXT: | `-DeclRefExpr {{.+}} 'int (struct opaque_s *__single)' Function {{.+}} 'f_opaque' 'int (struct opaque_s *__single)'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'struct opaque_s *__single' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-CStyleCastExpr {{.+}} 'struct opaque_s *__bidi_indexable' <BitCast>
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'unsigned char *__bidi_indexable' <ArrayToPointerDecay> part_of_explicit_cast
  // CHECK-NEXT:       `-CompoundLiteralExpr {{.+}} 'unsigned char[1]' lvalue
  // CHECK-NEXT:         `-InitListExpr {{.+}} 'unsigned char[1]'
  result = f_opaque((opaque_t)(unsigned char[1]){});

  // CHECK:      CallExpr {{.+}} 'int'
  // CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int (*__single)(void (*__single)(int))' <FunctionToPointerDecay>
  // CHECK-NEXT: | `-DeclRefExpr {{.+}} 'int (void (*__single)(int))' Function {{.+}} 'f_func' 'int (void (*__single)(int))'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'void (*__single)(int)' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-CStyleCastExpr {{.+}} 'void (*__bidi_indexable)(int)' <BitCast>
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'unsigned char *__bidi_indexable' <ArrayToPointerDecay> part_of_explicit_cast
  // CHECK-NEXT:       `-CompoundLiteralExpr {{.+}} 'unsigned char[1]' lvalue
  // CHECK-NEXT:         `-InitListExpr {{.+}} 'unsigned char[1]'
  result = f_func((void (*)(int))(unsigned char[1]){});

  // CHECK:      CallExpr {{.+}} 'int'
  // CHECK-NEXT: |-ImplicitCastExpr {{.+}} 'int (*__single)(__SVInt8_t *__single)' <FunctionToPointerDecay>
  // CHECK-NEXT: | `-DeclRefExpr {{.+}} 'int (__SVInt8_t *__single)' Function {{.+}} 'f_sizeless' 'int (__SVInt8_t *__single)'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} '__SVInt8_t *__single' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-CStyleCastExpr {{.+}} '__SVInt8_t *__bidi_indexable' <BitCast>
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'unsigned char *__bidi_indexable' <ArrayToPointerDecay> part_of_explicit_cast
  // CHECK-NEXT:       `-CompoundLiteralExpr {{.+}} 'unsigned char[1]' lvalue
  // CHECK-NEXT:         `-InitListExpr {{.+}} 'unsigned char[1]'
  result = f_sizeless((__SVInt8_t *)(unsigned char[1]){});
}
