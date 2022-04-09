// RUN: %clang_cc1 -no-opaque-pointers -fblocks -debug-info-kind=limited -emit-llvm %s -o - | FileCheck %s
// Make sure we do not generate line info for debugging-related frame setup.
// CHECK: define {{.*}}block_invoke
// CHECK-NOT: store {{.*}}%struct.__block_descriptor*{{.*}}dbg
// CHECK: store {{.*}}%struct.__block_descriptor*{{.*}}, align
// CHECK: ret
// CHECK: define {{.*}}block_invoke
// CHECK-NOT: store {{.*}}%struct.__block_descriptor*{{.*}}dbg
// CHECK: store {{.*}}%struct.__block_descriptor*{{.*}}, align
// CHECK: ret
// CHECK: define {{.*}}block_invoke
// CHECK-NOT: store {{.*}}%struct.__block_descriptor*{{.*}}dbg
// CHECK: store {{.*}}%struct.__block_descriptor*{{.*}}, align
// CHECK: ret
int printf(const char*, ...);

static void* _NSConcreteGlobalBlock;


typedef void (^ HelloBlock_t)(const char * name);

  /* Breakpoint for first Block function.  */
HelloBlock_t helloBlock = ^(const char * name) {
  printf("Hello there, %s!\n", name);
};

  /* Breakpoint for second Block function.  */
static HelloBlock_t s_helloBlock = ^(const char * name) {
  printf("Hello there, %s!\n", name);
};

/* Breakpoint for third Block function.  */
int X = 1234;
int (^CP)(void) = ^{ X = X+1;  return X; };

int
main(int argc, char * argv[])
{
  helloBlock("world");
  s_helloBlock("world");

  CP();
  printf ("X = %d\n", X);
  return X - 1235;
}
