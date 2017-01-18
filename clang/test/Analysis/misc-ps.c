// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -disable-free -analyzer-eagerly-assume -analyzer-checker=core,deadcode,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

int size_rdar9373039 = 1;
int foo_rdar9373039(const char *);

int rdar93730392() {
  int x;
  int j = 0;

  for (int i = 0 ; i < size_rdar9373039 ; ++i)
    x = 1;
    
  int extra = (2 + foo_rdar9373039 ("Clang") + ((4 - ((unsigned int) (2 + foo_rdar9373039 ("Clang")) % 4)) % 4)) + (2 + foo_rdar9373039 ("1.0") + ((4 - ((unsigned int) (2 + foo_rdar9373039 ("1.0")) % 4)) % 4)); // expected-warning {{never read}}

  for (int i = 0 ; i < size_rdar9373039 ; ++i)
    j += x; // expected-warning {{garbage}}

  return j;
}


int PR8962 (int *t) {
  // This should look through the __extension__ no-op.
  if (__extension__ (t)) return 0;
  return *t; // expected-warning {{null pointer}}
}

int PR8962_b (int *t) {
  // This should still ignore the nested casts
  // which aren't handled by a single IgnoreParens()
  if (((int)((int)t))) return 0;
  return *t; // expected-warning {{null pointer}}
}

int PR8962_c (int *t) {
  // If the last element in a StmtExpr was a ParenExpr, it's still live
  if (({ (t ? (_Bool)0 : (_Bool)1); })) return 0;
  return *t; // no-warning
}

int PR8962_d (int *t) {
  // If the last element in a StmtExpr is an __extension__, it's still live
  if (({ __extension__(t ? (_Bool)0 : (_Bool)1); })) return 0;
  return *t; // no-warning
}

int PR8962_e (int *t) {
  // Redundant casts can mess things up!
  // Environment used to skip through NoOp casts, but LiveVariables didn't!
  if (({ (t ? (int)(int)0L : (int)(int)1L); })) return 0;
  return *t; // no-warning
}

int PR8962_f (int *t) {
  // The StmtExpr isn't a block-level expression here,
  // the __extension__ is. But the value should be attached to the StmtExpr
  // anyway. Make sure the block-level check is /before/ IgnoreParens.
  if ( __extension__({
    _Bool r;
    if (t) r = 0;
    else r = 1;
    r;
  }) ) return 0;
  return *t; // no-warning
}

// This previously crashed logic in the analyzer engine when evaluating locations.
void rdar10308201_aux(unsigned val);
void rdar10308201 (int valA, void *valB, unsigned valC) {
  unsigned actual_base, lines;
  if (valC == 0) {
    actual_base = (unsigned)valB;
    for (;;) {
      if (valA & (1<<0))
        rdar10308201_aux(actual_base);
    }
  }
}

typedef struct Struct103 {
  unsigned i;
} Struct103;
typedef unsigned int size_t;
void __my_memset_chk(char*, int, size_t);
static int radar10367606(int t) {
  Struct103 overall;
  ((__builtin_object_size ((char *) &overall, 0) != (size_t) -1) ? __builtin___memset_chk ((char *) &overall, 0, sizeof(Struct103), __builtin_object_size ((char *) &overall, 0)) : __my_memset_chk ((char *) &overall, 0, sizeof(Struct103)));
  return 0;
}

/* Caching out on a sink node. */
extern int fooR10376675();
extern int* bazR10376675();
extern int nR10376675;
void barR10376675(int *x) {
  int *pm;
  if (nR10376675 * 2) {
    int *pk  = bazR10376675();
    pm = pk; //expected-warning {{never read}}
  }
  do {
    *x = fooR10376675();
  } while (0);
}

// Test accesses to wide character strings doesn't break the analyzer.
typedef int wchar_t;
struct rdar10385775 {
    wchar_t *name;
};
void RDar10385775(struct rdar10385775* p) {
    p->name = L"a";
}

// Test double loop of array and array literals.  Previously this
// resulted in a false positive uninitailized value warning.
void rdar10686586() {
    int array1[] = { 1, 2, 3, 0 };
    int array2[] = { 1, 2, 3, 0 };
    int *array[] = { array1, array2 };
    int sum = 0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            sum += array[i][j]; // no-warning
        }
    }
}

// This example tests CFG handling of '||' nested in a ternary expression,
// and seeing that the analyzer doesn't crash.
int isctype(char c, unsigned long f)
{
  return (c < 1 || c > 10) ? 0 : !!(c & f);
}

// Test that symbolic array offsets are modeled conservatively.
// This was triggering a false "use of uninitialized value" warning.
void rdar_12075238__aux(unsigned long y);
int rdar_12075238_(unsigned long count) {
  if ((count < 3) || (count > 6))
    return 0;
	
  unsigned long array[6];
  unsigned long i = 0;
  for (; i <= count - 2; i++)
  {
	  array[i] = i;
  }
  array[count - 1] = i;
  rdar_12075238__aux(array[2]); // no-warning
  return 0;
}

// Test that we handle an uninitialized value within a logical expression.
void PR14635(int *p) {
  int a = 0, b;
  *p = a || b; // expected-warning {{Assigned value is garbage or undefined}}
}

// Test handling floating point values with unary '!'.
int PR14634(int x) {
  double y = (double)x;
  return !y;
}


// PR15684: If a checker generates a sink node after generating a regular node
// and no state changes between the two, graph trimming would consider the two
// the same node, forming a loop.
struct PR15684 {
  void (*callback)(int);
};
void sinkAfterRegularNode(struct PR15684 *context) {
  int uninitialized;
  context->callback(uninitialized); // expected-warning {{uninitialized}}
}


// PR16131: C permits variables to be declared extern void.
static void PR16131(int x) {
  extern void v;

  int *ip = (int *)&v;
  char *cp = (char *)&v;
  clang_analyzer_eval(ip == cp); // expected-warning{{TRUE}}
  // expected-warning@-1 {{comparison of distinct pointer types}}

  *ip = 42;
  clang_analyzer_eval(*ip == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(*(int *)&v == 42); // expected-warning{{TRUE}}
}

// PR15623: Currently the analyzer doesn't handle symbolic expressions of the
// form "(exp comparison_op expr) != 0" very well. We perform a simplification
// translating an assume of a constraint of the form "(exp comparison_op expr)
// != 0" to true into an assume of "exp comparison_op expr" to true.
void PR15623(int n) {
  if ((n == 0) != 0) {
    clang_analyzer_eval(n == 0); // expected-warning{{TRUE}}
  }
}
