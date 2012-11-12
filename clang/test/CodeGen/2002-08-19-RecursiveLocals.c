// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

/* This testcase doesn't actually test a bug, it's just the result of me 
 * figuring out the syntax for forward declaring a static variable. */
struct list {
  int x;
  struct list *Next;
};

static struct list B;  /* Forward declare static */
static struct list A = { 7, &B };
static struct list B = { 8, &A };

extern struct list D;  /* forward declare normal var */

struct list C = { 7, &D };
struct list D = { 8, &C };

