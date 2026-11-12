// RUN: split-file %s %t
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c11 -emit-pch -o %t/unused.ast %t/unused.c
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c11 -emit-pch -o %t/values.ast %t/values.c
//
// DEFINE: %{analyze} = %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -std=c11 \
// DEFINE:   -analyzer-checker=core,debug.ExprInspection,deadcode.DeadStores \
// DEFINE:   -analyzer-config eagerly-assume=false,experimental-enable-naive-ctu-analysis=true,ctu-dir=%t,ctu-import-threshold=1,ctu-phase1-inlining=all
//
// Each entry runs in a fresh process, so earlier imports cannot hide a failure.
// RUN: %{analyze} -analyze-function=function_import -verify=callee %t/main.c
// RUN: %{analyze} -analyze-function=constants -verify=constants %t/main.c
// RUN: %{analyze} -analyze-function=array_extent -verify=extent %t/main.c
// RUN: %{analyze} -analyze-function=cfg_constants -verify=cfg %t/main.c
// RUN: %{analyze} -analyze-function=unevaluated -verify=unevaluated %t/main.c
// RUN: %{analyze} -analyze-function=through_callee -verify=limited %t/main.c
// RUN: %{analyze} -analyzer-config ctu-import-threshold=2 -analyze-function=through_callee -verify=dependency %t/main.c
// RUN: %{analyze} -analyzer-config ctu-import-threshold=0 -analyze-function=disabled -verify=disabled %t/main.c
// RUN: %{analyze} -analyzer-config experimental-enable-naive-ctu-analysis=false -analyze-function=disabled -verify=disabled %t/main.c

//--- main.c
void clang_analyzer_eval(int);
__SIZE_TYPE__ clang_analyzer_getExtent(const void *);
void observe(int);

// An unused header declaration must not consume the one available AST slot.
extern const int unused;
extern const int value;
extern const char bytes[];
struct Record {
  struct { int field; } nested;
};
extern const struct Record record;
int zero(void);
int read_dependency(void);
__SIZE_TYPE__ imported_extent(void);

int function_import(void) {
  return 7 / zero(); // callee-warning{{Division by zero}}
}

void constants(void) {
  clang_analyzer_eval(value == 7); // constants-warning{{TRUE}}
  clang_analyzer_eval(bytes[1] == 0); // constants-warning{{TRUE}}
  clang_analyzer_eval(record.nested.field == 5); // constants-warning{{TRUE}}
}

void array_extent(void) {
  // The definition is needed even before the first element is loaded.
  clang_analyzer_eval(clang_analyzer_getExtent(bytes) == 2); // extent-warning{{TRUE}}
}

void cfg_constants(void) {
  int scalar;
  scalar = 0; // cfg-warning{{Value stored to 'scalar' is never read}}
  if (value == 7)
    scalar = 1;
  observe(scalar);

  int field;
  field = 0; // cfg-warning{{Value stored to 'field' is never read}}
  if (record.nested.field == 5)
    field = 1;
  observe(field);
}

const int object_size = sizeof(unused);
int unevaluated(void) {
  observe(sizeof(unused));
  // Resolving initializer dependencies must also skip unevaluated expressions.
  clang_analyzer_eval(object_size == sizeof(int)); // unevaluated-warning{{TRUE}}
  return 7 / zero(); // unevaluated-warning{{Division by zero}}
}

void through_callee(void) {
  // The constant is only referenced in the imported function. Its definition
  // needs a second AST, and must still obey the shared import threshold.
  clang_analyzer_eval(read_dependency() == 7); // dependency-warning{{TRUE}} limited-warning{{UNKNOWN}}
  clang_analyzer_eval(imported_extent() == 2); // dependency-warning{{TRUE}} limited-warning{{UNKNOWN}}
}

void disabled(void) {
  clang_analyzer_eval(value == 7); // disabled-warning{{UNKNOWN}}
}

//--- unused.c
const int unused = 123;
const int dependency = 7;
const char hidden_bytes[] = "a";

//--- values.c
const int value = 7;
const char bytes[] = "a";
struct Record {
  struct { int field; } nested;
};
const struct Record record = {{5}};
int zero(void) { return 0; }
extern const int dependency;
int read_dependency(void) { return dependency; }
__SIZE_TYPE__ clang_analyzer_getExtent(const void *);
extern const char hidden_bytes[];
__SIZE_TYPE__ imported_extent(void) {
  return clang_analyzer_getExtent(hidden_bytes);
}

//--- externalDefMap.txt
9:c:@unused unused.ast
13:c:@dependency unused.ast
15:c:@hidden_bytes unused.ast
8:c:@value values.ast
8:c:@bytes values.ast
9:c:@record values.ast
9:c:@F@zero values.ast
20:c:@F@read_dependency values.ast
20:c:@F@imported_extent values.ast
