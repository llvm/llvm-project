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
enum B { x = 42,
         l,
         s };
int enumCheck(void) {
  return x;
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
int inlineAsm() {
  int res;
  asm("mov $42, %0"
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
