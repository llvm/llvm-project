

// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -fbounds-safety %s 2>&1 \
// RUN:   | FileCheck %s

#include <ptrcheck.h>

// BoundsSafetyPointerPromotionExpr

// CHECK-LABEL: int *__bidi_indexablepromote_counted_by(int *ptr, unsigned int len)
// CHECK: [B2 (ENTRY)]
// CHECK-NEXT:   Succs (1): B1
// CHECK: [B1]
// CHECK-NEXT:   1: ptr
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, LValueToRValue, int *__single __counted_by(len))
// CHECK-NEXT:   3: len
// CHECK-NEXT:   4: [B1.3] (ImplicitCastExpr, LValueToRValue, unsigned int)
// CHECK-NEXT:   5: [B1.2] (BoundsSafetyPointerPromotionExpr, int *__bidi_indexable)
// CHECK-NEXT:   6: return [B1.5];
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
// CHECK: [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
int *__bidi_indexable
promote_counted_by(int *__counted_by(len) ptr, unsigned len) {
  return ptr;
}

// BoundsCheckExpr

typedef struct {
  unsigned long long size;
  void *__sized_by(size) buf;
} counted_buf;

// CHECK-LABEL: void bounds_check(counted_buf *bf, void *__bidi_indexablebuf, unsigned long long size)
// CHECK: [B2 (ENTRY)]
// CHECK-NEXT:   Succs (1): B1
// CHECK: [B1]
// CHECK-NEXT:   1: buf
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, LValueToRValue, void *__bidi_indexable)
// CHECK-NEXT:   3: size
// CHECK-NEXT:   4: [B1.3] (ImplicitCastExpr, LValueToRValue, unsigned long long)
// CHECK-NEXT:   5: [B1.2] (ImplicitCastExpr, BoundsSafetyPointerCast, void *__single __sized_by(size))
// CHECK-NEXT:   6: bf
// CHECK-NEXT:   7: [B1.6] (ImplicitCastExpr, LValueToRValue, counted_buf *__single)
// CHECK-NEXT:   8: [B1.7]->buf
// CHECK-NEXT:   9: [B1.8] = [B1.5]
// CHECK-NEXT:  10: bf
// CHECK-NEXT:  11: [B1.10] (ImplicitCastExpr, LValueToRValue, counted_buf *__single)
// CHECK-NEXT:  12: [B1.11]->size
// CHECK-NEXT:  13: [B1.12] = [B1.4]
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
// CHECK: [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void bounds_check(counted_buf *bf,
                  void *__bidi_indexable buf,
                  unsigned long long size) {
  bf->buf = buf;
  bf->size = size;
}

// Combined

void *__sized_by(size) my_alloc(unsigned long long size);

// CHECK-LABEL: int promote_and_bounds_check_in_middle(int v, counted_buf *bf)
// CHECK:  [B6 (ENTRY)]
// CHECK-NEXT:    Succs (1): B5
// CHECK:  [B1]
// CHECK-NEXT:    1: v
// CHECK-NEXT:    2: [B1.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:    3: return [B1.2];
// CHECK-NEXT:    Preds (1): B3
// CHECK-NEXT:    Succs (1): B0
// CHECK:  [B2]
// CHECK-NEXT:    1: 9
// CHECK-NEXT:    2: return [B2.1];
// CHECK-NEXT:    Preds (1): B3
// CHECK-NEXT:    Succs (1): B0
// CHECK:  [B3]
// CHECK-NEXT:    1: 10
// CHECK-NEXT:    2: [B3.1] (ImplicitCastExpr, IntegralCast, unsigned long long)
// CHECK-NEXT:    3: my_alloc
// CHECK-NEXT:    4: [B3.3] (ImplicitCastExpr, FunctionToPointerDecay, void *__single __sized_by(size)(*__single)(unsigned long long))
// CHECK-NEXT:    5: [B3.4]([B3.2])
// CHECK-NEXT:    6: [B3.5] (BoundsSafetyPointerPromotionExpr, void *__bidi_indexable)
// CHECK-NEXT:    7: 8
// CHECK-NEXT:    8: [B3.7] (ImplicitCastExpr, IntegralCast, unsigned long long)
// CHECK-NEXT:    9: [B3.6] (ImplicitCastExpr, BoundsSafetyPointerCast, void *__single __sized_by(size))
// CHECK-NEXT:   10: bf
// CHECK-NEXT:   11: [B3.10] (ImplicitCastExpr, LValueToRValue, counted_buf *__single)
// CHECK-NEXT:   12: [B3.11]->buf
// CHECK-NEXT:   13: [B3.12] = [B3.9]
// CHECK-NEXT:   14: bf
// CHECK-NEXT:   15: [B3.14] (ImplicitCastExpr, LValueToRValue, counted_buf *__single)
// CHECK-NEXT:   16: [B3.15]->size
// CHECK-NEXT:   17: [B3.16] = [B3.8]
// CHECK-NEXT:   18: v
// CHECK-NEXT:   19: [B3.18] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   20: 9
// CHECK-NEXT:   21: [B3.19] < [B3.20]
// CHECK-NEXT:    T: if [B3.21]
// CHECK-NEXT:    Preds (1): B5
// CHECK-NEXT:    Succs (2): B2 B1
// CHECK:  [B4]
// CHECK-NEXT:    1: 0
// CHECK-NEXT:    2: return [B4.1];
// CHECK-NEXT:    Preds (1): B5
// CHECK-NEXT:    Succs (1): B0
// CHECK:  [B5]
// CHECK-NEXT:    1: v
// CHECK-NEXT:    2: [B5.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:    3: ![B5.2]
// CHECK-NEXT:    T: if [B5.3]
// CHECK-NEXT:    Preds (1): B6
// CHECK-NEXT:    Succs (2): B4 B3
// CHECK:  [B0 (EXIT)]
// CHECK-NEXT:    Preds (3): B1 B2 B4
int promote_and_bounds_check_in_middle(int v, counted_buf *bf) {
  if (!v)
    return 0;

  bf->buf = my_alloc(10);
  bf->size = 8;

  if (v < 9)
    return 9;

  return v;
}


// example from Sema/unreachable-noret.c

#define NO_RETURN __attribute__((noreturn))
void NO_RETURN halt(const void * const p_fatal_error);

// CHECK-LABEL: static void handler_private(const void *p_stack)
// CHECK: [B2 (ENTRY)]
// CHECK-NEXT:   Succs (1): B1
// CHECK: [B1 (NORETURN)]
// CHECK-NEXT:   1: int foo;
// CHECK-NEXT:   2: halt
// CHECK-NEXT:   3: [B1.2] (ImplicitCastExpr, FunctionToPointerDecay, void (*__single)(const void *__singleconst) __attribute__((noreturn)))
// CHECK-NEXT:   4: foo
// CHECK-NEXT:   5: &[B1.4]
// CHECK-NEXT:   6: [B1.5] (ImplicitCastExpr, BitCast, const void *__bidi_indexable)
// CHECK-NEXT:   7: [B1.6] (ImplicitCastExpr, BoundsSafetyPointerCast, const void *__single)
// CHECK-NEXT:   8: [B1.3]([B1.7])
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
// CHECK: [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
static void NO_RETURN handler_private(const void *__sized_by(0x78) p_stack)
{
  int foo;
  halt(&foo);
}

// CHECK-LABEL: void handler_irq(const void *p_stack)
// CHECK: [B2 (ENTRY)]
// CHECK-NEXT:   Succs (1): B1
// CHECK: [B1 (NORETURN)]
// CHECK-NEXT:   1: p_stack
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, LValueToRValue, const void *__single __sized_by(120))
// CHECK-NEXT:   3: 120
// CHECK-NEXT:   4: [B1.2] (BoundsSafetyPointerPromotionExpr, const void *__bidi_indexable)
// CHECK-NEXT:   5: 120
// CHECK-NEXT:   6: [B1.5] (ImplicitCastExpr, IntegralCast, long)
// CHECK-NEXT:   7: handler_private
// CHECK-NEXT:   8: [B1.7] (ImplicitCastExpr, FunctionToPointerDecay, void (*__single)(const void *__single __sized_by(120)) __attribute__((noreturn)))
// CHECK-NEXT:   9: [B1.4] (ImplicitCastExpr, BoundsSafetyPointerCast, const void *__single __sized_by(120))
// CHECK-NEXT:  10: [B1.8]([B1.9])
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
// CHECK: [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void NO_RETURN handler_irq(const void *__sized_by(0x78) p_stack)
{
  handler_private(p_stack);
}
