// RUN: %clang_cc1 %s -fobjc-arc -S -emit-llvm -o /dev/null

// Don't crash if the argument type and the parameter type in an indirect copy
// restore expression have different qualification.
@protocol P1
@end

typedef int handler(id<P1> *const p);

int main(void) {
  id<P1> i1 = 0;
  handler *func = 0;
  return func(&i1);
}
