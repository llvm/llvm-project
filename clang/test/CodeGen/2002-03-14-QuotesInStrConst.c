// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

/* GCC was not escaping quotes in string constants correctly, so this would
 * get emitted:
 *  %.LC1 = internal global [32 x sbyte] c"*** Word "%s" on line %d is not\00"
 */

const char *Foo() {
  return "*** Word \"%s\" on line %d is not";
}
