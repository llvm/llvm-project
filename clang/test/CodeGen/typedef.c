// RUN: %clang_cc1 -emit-llvm %s -o -

typedef struct { int i; } Value;
typedef Value *PValue;

int get_value(PValue v) {
  return v->i;
}
