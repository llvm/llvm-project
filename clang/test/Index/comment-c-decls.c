// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng -target x86_64-apple-darwin10 %s > %t/out
// RUN: FileCheck %s < %t/out

// Ensure that XML we generate is not invalid.
// RUN: FileCheck %s -check-prefix=WRONG < %t/out
// WRONG-NOT: CommentXMLInvalid
// rdar://12378714

/**
 * \brief Aaa.
*/
int global_function();
// CHECK: <Declaration>int global_function()</Declaration>

/**
 * \param x1 Aaa.
*/
extern void external_function(int x1);
// CHECK: <Declaration>extern void external_function(int x1)</Declaration>

/**
 * \brief global variable;
*/
int global_variable;
// CHECK: <Declaration>int global_variable</Declaration>

/**
 * \brief local variable;
*/
static int static_variable;
// CHECK: <Declaration>static int static_variable</Declaration>

/**
 * \brief external variable
*/
extern int external_variable;
// CHECK: <Declaration>extern int external_variable</Declaration>

int global_function() {
  /**
   * \brief a local variable
  */
  int local = 10;
  return local;
}
// CHECK: <Declaration>int global_function()</Declaration>
// CHECK: <Declaration>int local = 10</Declaration>

/**
 * \brief initialized decl.
*/
int initialized_global = 100;
// CHECK: <Declaration>int initialized_global = 100</Declaration>

/**
 * \brief typedef example
*/
typedef int INT_T;
// CHECK: <Declaration>typedef int INT_T</Declaration>

/**
 * \brief aggregate type example
*/
struct S {
/**
 * \brief iS1;
*/
  int iS1;
/**
 * \brief dS1;
*/
  double dS1;
};
// CHECK: <Declaration>struct S {}</Declaration>
// CHECK: <Declaration>int iS1</Declaration>
// CHECK: <Declaration>double dS1</Declaration>

/**
 * \brief enum e;
*/
enum e {
  One,
/**
 * \brief Two;
*/
  Two,
  Three
};
// CHECK: <Declaration>enum e {}</Declaration>
// CHECK: <Declaration>Two</Declaration>

/**
 *\brief block declaration
*/
int (^Block) (int i, int j);
// CHECK: <Declaration>int (^Block)(int, int)</Declaration>

/**
 *\brief block declaration
*/
int (^Block1) (int i, int j) = ^(int i, int j) { return i + j; };
// CHECK: <Declaration>int (^Block1)(int, int) = ^(int i, int j) {\n}</Declaration>
