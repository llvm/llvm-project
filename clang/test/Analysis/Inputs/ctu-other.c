// Test typedef and global variable in function.
typedef struct {
  int a;
  int b;
} FooBar;
FooBar fb;
int f(int i) {
  if (fb.a) {
    fb.b = i;
  }
  return 1;
}

// Test enums.
enum B { x2 = 42,
         y2,
         z2 };
int enumCheck(void) {
  return x2;
}

// Test reporting an error in macro definition
#define MYMACRO(ctx) \
  ctx->a;
struct S {
  int a;
};
int g(struct S *ctx) {
  MYMACRO(ctx);
  return 0;
}

// Test that asm import does not fail.
// TODO: Support the GNU extension asm keyword as well.
// Example using the GNU extension: asm("mov $42, %0" : "=r"(res));
int inlineAsm() {
  int res;
  __asm__("mov $42, %0"
          : "=r"(res));
  return res;
}

// Implicit function.
int identImplicit(int in) {
  return in;
}

// ASTImporter doesn't support this construct.
int structInProto(struct DataType {int a;int b; } * d) {
  return 0;
}
