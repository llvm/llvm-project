// Simple declaration rewrite end-to-end test from source code to
// applied replacements.

// RUN: rm -rf %t && mkdir -p %t
// RUN: split-file %s %t

// DEFINE: %{testname} = unset
// DEFINE: %{apply_cpp} = %t/%{testname}_apply/%{testname}.cpp
// DEFINE: %{orig_cpp} = %t/%{testname}_apply/%{testname}.orig.cpp
// DEFINE: %{edits_yaml} = %t/%{testname}_apply/%{testname}.edits.yaml
// DEFINE: %{report_sarif} = %t/%{testname}.report.sarif
// DEFINE: %{extract} = %clang -c %t/%{testname}.cpp -o %t/%{testname}.o \
// DEFINE:   --ssaf-extract-summaries=PointerFlow,UnsafeBufferUsage \
// DEFINE:   --ssaf-compilation-unit-id=%{testname}.cu \
// DEFINE:   --ssaf-tu-summary-file=%t/%{testname}.tu.json
// DEFINE: %{link} = clang-ssaf-linker %t/%{testname}.tu.json -o %t/%{testname}.lu.json
// DEFINE: %{analyze} = clang-ssaf-analyzer %t/%{testname}.lu.json \
// DEFINE:   -o %t/%{testname}.wpa.json -a UnsafeBufferReachableAnalysisResult
// DEFINE: %{make_apply_copy} = mkdir -p %t/%{testname}_apply && \
// DEFINE:   sed '/^\/\/ REDEFINE:/,$d' %t/%{testname}.cpp > %{apply_cpp} && \
// DEFINE:   cp %{apply_cpp} %{orig_cpp}
// DEFINE: %{transform} = %clang -c %{apply_cpp} -o %t/%{testname}.test2.o \
// DEFINE:   --ssaf-source-transformation=cpp-bounded-buffers \
// DEFINE:   --ssaf-global-scope-analysis-result=%t/%{testname}.wpa.json \
// DEFINE:   --ssaf-src-edit-file=%{edits_yaml} \
// DEFINE:   --ssaf-transformation-report-file=%{report_sarif} \
// DEFINE:   --ssaf-compilation-unit-id=%{testname}.cu \
// DEFINE:   --ssaf-link-unit-id=%{testname}.lu
// DEFINE: %{apply} = clang-apply-replacements %t/%{testname}_apply


//--- void_buffer_ptr.cpp
void use(void *p) {
  ((char *)p)[5];
}

// REDEFINE: %{testname} = void_buffer_ptr
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: %{apply}
// RUN: FileCheck --check-prefix=VOID_BUFFER_PTR --input-file=%{apply_cpp} %s
// VOID_BUFFER_PTR: void use(bounded_ptr<char> p)


//--- pointer_global.cpp
int *p;
void use() { p[5] = 0; }

// REDEFINE: %{testname} = pointer_global
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: %{apply}
// RUN: FileCheck --check-prefix=POINTER_GLOBAL_REWRITTEN --input-file=%{apply_cpp} %s
// POINTER_GLOBAL_REWRITTEN: bounded_ptr<int> p;


//--- specifier_static.cpp
static int *p;
void use() { p[5] = 0; }
// REDEFINE: %{testname} = specifier_static
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: %{apply}
// RUN: FileCheck --check-prefix=SPECIFIER_STATIC_REWRITTEN --input-file=%{apply_cpp} %s
// SPECIFIER_STATIC_REWRITTEN: static bounded_ptr<int> p;

// ============================================================================
// Qualifiers, and their positions relative to the pointee/element type
// ============================================================================


//--- qualifier_const_leading.cpp
const char *p;
void use() { (void)p[5]; }

// REDEFINE: %{testname} = qualifier_const_leading
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: %{apply}
// RUN: FileCheck --check-prefix=QUALIFIER_CONST_LEADING_REWRITTEN --input-file=%{apply_cpp} %s
// QUALIFIER_CONST_LEADING_REWRITTEN: bounded_ptr<const char> p;


//--- qualifier_const_trailing_spelled_after.cpp
char const *p;
void use() {
  (void)p[5];
}

// REDEFINE: %{testname} = qualifier_const_trailing_spelled_after
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: %{apply}
// RUN: FileCheck --check-prefix=QUALIFIER_SPELLED_AFTER_REWRITTEN --input-file=%{apply_cpp} %s
// QUALIFIER_SPELLED_AFTER_REWRITTEN: bounded_ptr<const char> p;


//--- qualifier_on_pointer_itself.cpp
int *const p = nullptr;
void use() { (void)p[5]; }

// REDEFINE: %{testname} = qualifier_on_pointer_itself
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: %{apply}
// RUN: FileCheck --check-prefix=QUALIFIER_ON_POINTER_REWRITTEN --input-file=%{apply_cpp} %s
// QUALIFIER_ON_POINTER_REWRITTEN: bounded_ptr<int> const p = nullptr;


//--- qualifier_multiple_trailing.cpp
int *volatile const p = nullptr;
void use() { (void)p[5]; }

// REDEFINE: %{testname} = qualifier_multiple_trailing
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: %{apply}
// RUN: FileCheck --check-prefix=QUALIFIER_MULTIPLE_TRAILING_REWRITTEN --input-file=%{apply_cpp} %s
// QUALIFIER_MULTIPLE_TRAILING_REWRITTEN: bounded_ptr<int> volatile const p = nullptr;


//--- qualifier_array_element_const.cpp
const int arr[3] = {};
void use() { (void)arr[5]; }

// REDEFINE: %{testname} = qualifier_array_element_const
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: %{apply}
// RUN: FileCheck --check-prefix=QUALIFIER_ARRAY_ELEMENT_REWRITTEN --input-file=%{apply_cpp} %s
// QUALIFIER_ARRAY_ELEMENT_REWRITTEN: bounded_array<const int, 3> arr = {};


//--- qualifier_array_multiple_trailing_reversed.cpp
int volatile const arr[3] = {};
void use() {
  (void)arr[5];
}

// REDEFINE: %{testname} = qualifier_array_multiple_trailing_reversed
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: %{apply}
// RUN: FileCheck --check-prefix=QUALIFIER_ARRAY_REVERSED_REWRITTEN --input-file=%{apply_cpp} %s
// QUALIFIER_ARRAY_REVERSED_REWRITTEN: bounded_array<const volatile int, 3> arr = {};


//--- qualifier_leading_separated.cpp
const static char *p;
void use() {
  (void)p[5];
}

// REDEFINE: %{testname} = qualifier_leading_separated
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=QUALIFIER_LEADING_SEPARATED --input-file=%{edits_yaml} %s
// RUN: FileCheck --check-prefix=QUALIFIER_LEADING_SEPARATED_REPORT --input-file=%{report_sarif} %s
// QUALIFIER_LEADING_SEPARATED: Replacements: []
// QUALIFIER_LEADING_SEPARATED_REPORT: "text": "unexpected token between a leading cv-qualifier and the type"
// RUN: %{apply}
// RUN: diff %{orig_cpp} %{apply_cpp}


//--- qualifier_trailing_separated.cpp
int /* c */ const arr[3] = {};
void use() {
  arr[5];
}

// REDEFINE: %{testname} = qualifier_trailing_separated
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=QUALIFIER_TRAILING_SEPARATED --input-file=%{edits_yaml} %s
// RUN: FileCheck --check-prefix=QUALIFIER_TRAILING_SEPARATED_REPORT --input-file=%{report_sarif} %s
// QUALIFIER_TRAILING_SEPARATED: Replacements: []
// QUALIFIER_TRAILING_SEPARATED_REPORT: "text": "unexpected token between the type and a trailing cv-qualifier"
// RUN: %{apply}
// RUN: diff %{orig_cpp} %{apply_cpp}

//--- array_plain.cpp
void use() {
  int arr[3];
  arr[5] = 0;
}

// REDEFINE: %{testname} = array_plain
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: %{apply}
// RUN: FileCheck --check-prefix=ARRAY_PLAIN_REWRITTEN --input-file=%{apply_cpp} %s
// ARRAY_PLAIN_REWRITTEN: bounded_array<int, 3> arr;

// An array of pointers: the element type is itself a pointer, but the
// element is not dereferenced by the array rewrite, so it is reproduced
// verbatim inside the angle brackets.
//--- array_of_pointers.cpp
void use() {
  int *arr[3];
  arr[5] = nullptr;
}

// REDEFINE: %{testname} = array_of_pointers
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: %{apply}
// RUN: FileCheck --check-prefix=ARRAY_OF_POINTERS_REWRITTEN --input-file=%{apply_cpp} %s
// ARRAY_OF_POINTERS_REWRITTEN: bounded_array<int *, 3>arr;


//--- array_multi_dim.cpp
void use() {
  int arr[3][4];
  arr[5][0] = 0;
}

// REDEFINE: %{testname} = array_multi_dim
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=ARRAY_MULTI_DIM --input-file=%{edits_yaml} %s
// RUN: FileCheck --check-prefix=ARRAY_MULTI_DIM_REPORT --input-file=%{report_sarif} %s
// ARRAY_MULTI_DIM: Replacements: []
// ARRAY_MULTI_DIM_REPORT: "text": "multi-dimensional array is not yet rewritten"
// RUN: %{apply}
// RUN: diff %{orig_cpp} %{apply_cpp}


//--- array_incomplete.cpp
extern int arr[];
void use() {
  arr[5] = 0;
}

// REDEFINE: %{testname} = array_incomplete
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=ARRAY_INCOMPLETE --input-file=%{edits_yaml} %s
// RUN: FileCheck --check-prefix=ARRAY_INCOMPLETE_REPORT --input-file=%{report_sarif} %s
// ARRAY_INCOMPLETE: Replacements: []
// ARRAY_INCOMPLETE_REPORT: "text": "array of unknown bound is not yet rewritten"
// RUN: %{apply}
// RUN: diff %{orig_cpp} %{apply_cpp}


//--- array_of_function_pointers_raw.cpp
void use() {
  void (*arr[4])();
  arr[5]();
}

// REDEFINE: %{testname} = array_of_function_pointers_raw
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=ARRAY_RAW_FNPTR --input-file=%{edits_yaml} %s
// RUN: FileCheck --check-prefix=ARRAY_RAW_FNPTR_REPORT --input-file=%{report_sarif} %s
// ARRAY_RAW_FNPTR: Replacements: []
// ARRAY_RAW_FNPTR_REPORT: "text": "the array type does not end in a closing bracket"
// RUN: %{apply}
// RUN: diff %{orig_cpp} %{apply_cpp}


//--- pointer_local_alias.cpp
void use(int *p) {
  int *q = p;
  q[5] = 0;
}

// REDEFINE: %{testname} = pointer_local_alias
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: %{apply}
// RUN: FileCheck --check-prefix=POINTER_LOCAL_ALIAS_REWRITTEN --input-file=%{apply_cpp} %s
// POINTER_LOCAL_ALIAS_REWRITTEN: void use(bounded_ptr<int> p) {
// POINTER_LOCAL_ALIAS_REWRITTEN-NEXT: bounded_ptr<int> q = p;

// A reachable function return value.
//--- pointer_return_value.cpp
int *get(int *p) {
  p[5] = 0;
  return p;
}

// REDEFINE: %{testname} = pointer_return_value
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: %{apply}
// RUN: FileCheck --check-prefix=POINTER_RETURN_VALUE_REWRITTEN --input-file=%{apply_cpp} %s
// POINTER_RETURN_VALUE_REWRITTEN: int *get(bounded_ptr<int> p) {


//--- pointer_multi_level.cpp
void use(int **pp) {
  pp[5] = nullptr;
  (*pp)[5] = 0;
}

// REDEFINE: %{testname} = pointer_multi_level
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=POINTER_MULTI_LEVEL --input-file=%{edits_yaml} %s
// RUN: FileCheck --check-prefix=POINTER_MULTI_LEVEL_REPORT --input-file=%{report_sarif} %s
// POINTER_MULTI_LEVEL: Replacements: []
// POINTER_MULTI_LEVEL_REPORT: "text": "multi-level pointer indirection is not yet rewritten"
// RUN: %{apply}
// RUN: diff %{orig_cpp} %{apply_cpp}

//--- pointer_to_array.cpp
void use(int (*p)[3]) {
  p[5][0] = 0;
}

// REDEFINE: %{testname} = pointer_to_array
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=POINTER_TO_ARRAY --input-file=%{edits_yaml} %s
// RUN: FileCheck --check-prefix=POINTER_TO_ARRAY_REPORT --input-file=%{report_sarif} %s
// POINTER_TO_ARRAY: Replacements: []
// POINTER_TO_ARRAY_REPORT: "text": "pointer to array is not yet rewritten"
// RUN: %{apply}
// RUN: diff %{orig_cpp} %{apply_cpp}


//--- reference_to_pointer.cpp
void use(int *&p) {
  p[5] = 0;
}

// REDEFINE: %{testname} = reference_to_pointer
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=REFERENCE_TO_POINTER --input-file=%{edits_yaml} %s
// RUN: FileCheck --check-prefix=REFERENCE_TO_POINTER_REPORT --input-file=%{report_sarif} %s
// REFERENCE_TO_POINTER: Replacements: []
// REFERENCE_TO_POINTER_REPORT: "text": "reference to pointer is not yet rewritten"
// RUN: %{apply}
// RUN: diff %{orig_cpp} %{apply_cpp}


//--- pointer_parenthesized_declarator.cpp
void use() {
  int v = 0;
  int (*p) = &v;
  (void)p[5];
}

// REDEFINE: %{testname} = pointer_parenthesized_declarator
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=POINTER_PARENTHESIZED --input-file=%{edits_yaml} %s
// RUN: FileCheck --check-prefix=POINTER_PARENTHESIZED_REPORT --input-file=%{report_sarif} %s
// POINTER_PARENTHESIZED: Replacements: []
// POINTER_PARENTHESIZED_REPORT: "text": "pointer declarator does not end at its '*'"
// RUN: %{apply}
// RUN: diff %{orig_cpp} %{apply_cpp}


//--- field_pointer.cpp
struct S{int *p;};
void use(S *w) {
  w->p[5] = 0;
}

// REDEFINE: %{testname} = field_pointer
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: %{apply}
// RUN: FileCheck --check-prefix=FIELD_POINTER_REWRITTEN --input-file=%{apply_cpp} %s
// FIELD_POINTER_REWRITTEN: struct S{bounded_ptr<int> p;};

// An array-typed struct field.
//--- field_array.cpp
struct S{int arr[3];};
void use(S *w) {
  w->arr[5] = 0;
}

// REDEFINE: %{testname} = field_array
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: %{apply}
// RUN: FileCheck --check-prefix=FIELD_ARRAY_REWRITTEN --input-file=%{apply_cpp} %s
// FIELD_ARRAY_REWRITTEN: struct S{bounded_array<int, 3> arr;};

// ============================================================================
// Macros
// ============================================================================

// Skip MacroExpansion: the declarator's type is spelled through a macro.
//--- macro_expansion.cpp
#define PTR int *
void use(PTR p) {
  p[5] = 0;
}

// REDEFINE: %{testname} = macro_expansion
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=MACRO_EXPANSION --input-file=%{edits_yaml} %s
// RUN: FileCheck --check-prefix=MACRO_EXPANSION_REPORT --input-file=%{report_sarif} %s
// MACRO_EXPANSION: Replacements: []
// MACRO_EXPANSION_REPORT: "text": "declarator spelled through a macro is not yet rewritten"
// RUN: %{apply}
// RUN: diff %{orig_cpp} %{apply_cpp}

// ============================================================================
// Typedefs
// ============================================================================

// A typedef used as an array element type does not block the array rewrite:
// the typedef keeps the declarator a clean prefix + [N] suffix, so only the
// (unexpanded) element spelling changes.
//--- typedef_array_of_function_pointers.cpp
typedef void (*FP)();
void use() {
  FP arr[4];
  arr[5]();
}

// REDEFINE: %{testname} = typedef_array_of_function_pointers
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: %{apply}
// RUN: FileCheck --check-prefix=TYPEDEF_ARRAY_OF_FNPTRS_REWRITTEN --input-file=%{apply_cpp} %s
// TYPEDEF_ARRAY_OF_FNPTRS_REWRITTEN: bounded_array<FP, 4> arr;


//--- typedef_pointer.cpp
typedef int *P;
void use(P p) {
  p[5] = 0;
}

// REDEFINE: %{testname} = typedef_pointer
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=TYPEDEF_POINTER --input-file=%{edits_yaml} %s
// RUN: FileCheck --check-prefix=TYPEDEF_POINTER_REPORT --input-file=%{report_sarif} %s
// TYPEDEF_POINTER: Replacements: []
// TYPEDEF_POINTER_REPORT: "text": "no TypeLoc for the pointee or array element type"
// RUN: %{apply}
// RUN: diff %{orig_cpp} %{apply_cpp}


//--- unnamable_anonymous_struct.cpp
struct { int x; } *p;
void use(int i) {
  p[i].x = 0;
}

// REDEFINE: %{testname} = unnamable_anonymous_struct
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=UNNAMABLE_ANON_STRUCT --input-file=%{edits_yaml} %s
// RUN: FileCheck --check-prefix=UNNAMABLE_ANON_STRUCT_REPORT --input-file=%{report_sarif} %s
// UNNAMABLE_ANON_STRUCT: Replacements: []
// UNNAMABLE_ANON_STRUCT_REPORT: "text": "the pointee or array element type has no name that can be written as a template argument"
// RUN: %{apply}
// RUN: diff %{orig_cpp} %{apply_cpp}


//--- unnamable_lambda_decltype.cpp
void use() {
  auto f = [](int x) { return x; };
  decltype(f) *p = &f;
  (void)p[5];
}

// REDEFINE: %{testname} = unnamable_lambda_decltype
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=UNNAMABLE_LAMBDA --input-file=%{edits_yaml} %s
// RUN: FileCheck --check-prefix=UNNAMABLE_LAMBDA_REPORT --input-file=%{report_sarif} %s
// UNNAMABLE_LAMBDA: Replacements: []
// UNNAMABLE_LAMBDA_REPORT: "text": "the pointee or array element type has no name that can be written as a template argument"
// RUN: %{apply}
// RUN: diff %{orig_cpp} %{apply_cpp}


//--- whitespace_free_pointer.cpp
int*p;
void use(){p[5]=0;}

// REDEFINE: %{testname} = whitespace_free_pointer
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: %{apply}
// RUN: FileCheck --check-prefix=WHITESPACE_FREE_POINTER_REWRITTEN --input-file=%{apply_cpp} %s
// WHITESPACE_FREE_POINTER_REWRITTEN: bounded_ptr<int> p;


//--- skip_declaration_group.cpp
void use() {
  int *a, *b;
  a[5] = 0;
  b[5] = 0;
}

// REDEFINE: %{testname} = skip_declaration_group
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=SKIP_DECL_GROUP --input-file=%{edits_yaml} %s
// RUN: FileCheck --check-prefix=SKIP_DECL_GROUP_REPORT --input-file=%{report_sarif} %s
// SKIP_DECL_GROUP: Replacements: []
// SKIP_DECL_GROUP_REPORT-DAG: "text": "declarator of a multi-declarator group is not yet rewritten"
// SKIP_DECL_GROUP_REPORT-DAG: "text": "declarator of a multi-declarator group is not yet rewritten"
// RUN: %{apply}
// RUN: diff %{orig_cpp} %{apply_cpp}


//--- skip_trailing_return_type.cpp
auto f(int *p) -> int * {
  return p;
}
void use() {
  int *q = f(nullptr);
  q[5] = 0;
}

// REDEFINE: %{testname} = skip_trailing_return_type
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{make_apply_copy}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=SKIP_TRAILING_RETURN_REPORT --input-file=%{report_sarif} %s
// SKIP_TRAILING_RETURN_REPORT: "text": "trailing return type is not yet rewritten"
// RUN: %{apply}
// RUN: FileCheck --check-prefix=SKIP_TRAILING_RETURN_REWRITTEN --input-file=%{apply_cpp} %s
// SKIP_TRAILING_RETURN_REWRITTEN: auto f(bounded_ptr<int> p) -> int * {
// SKIP_TRAILING_RETURN_REWRITTEN: bounded_ptr<int> q = f(nullptr);
