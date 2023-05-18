// RUN: %clang_cc1 -emit-llvm -o - %s

/* WG14 N1311: Yes
 * Initializing static or external variables
 */

static int x;
static union {
  void *vp;
  float f;
  int i;
} u;

int main(void) {
  return x + u.i;
}

// CHECK: @x ={{.*}}i32 0
// CHECK-NEXT: @u ={{.*}}zeroinitializer
