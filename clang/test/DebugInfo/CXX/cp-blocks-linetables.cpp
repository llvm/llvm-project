// RUN: %clang_cc1 -fblocks -debug-info-kind=limited -emit-llvm %s -o - | FileCheck %s
// Ensure that we generate a line table entry for the block cleanup.
// CHECK: define {{.*}} @__main_block_invoke
// CHECK: _NSConcreteStackBlock
// CHECK: call {{.*}} @_Block_object_dispose{{.*}}, !dbg ![[L1:[0-9]+]]
// CHECK: ret

void * _NSConcreteStackBlock;
#ifdef __cplusplus
extern "C" void exit(int);
#else
extern void exit(int);
#endif

enum numbers {
  zero, one, two, three, four
};

typedef enum numbers (^myblock)(enum numbers);


double test(myblock I) {
  return I(three);
}

int main() {
  __block enum numbers x = one;
  __block enum numbers y = two;

  /* Breakpoint for first Block function.  */
  myblock CL = ^(enum numbers z)
    { enum numbers savex = x;
      { __block enum numbers x = savex;
	y = z;
	if (y != three)
	  exit (6);
	test (
	      /* Breakpoint for second Block function.  */
	      ^ (enum numbers z) {
		if (y != three) {
		  exit(1);
		}
		if (x != one)
		  exit(2);
		x = z;
		if (x != three)
		  exit(3);
		if (y != three)
		  exit(4);
		return (enum numbers) four;
	      });}
      return x;
    };

  enum numbers res = (enum numbers)test(CL);

  if (res != one)
    exit (5);
  return 0;
}
