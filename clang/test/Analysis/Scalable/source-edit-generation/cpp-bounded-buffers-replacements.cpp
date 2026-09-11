// A simple end-to-end test for declaration rewriting, from the source
// code to the generated source-edit YAML. If the
// 'clang-apply-replacements' tool is available, it also applies the
// YAML and checks the resulting rewritten source.


// RUN: rm -rf %t && mkdir -p %t
// RUN: split-file %s %t

// DEFINE: %{testname} = unset
// DEFINE: %{apply_dir} = %t/%{testname}_apply
// DEFINE: %{edits_yaml} = %{apply_dir}/%{testname}.edits.yaml
// DEFINE: %{report_sarif} = %t/%{testname}.report.sarif
// DEFINE: %{extract} = %clang -fsyntax-only %t/%{testname}.cpp \
// DEFINE:   --ssaf-extract-summaries=PointerFlow,UnsafeBufferUsage \
// DEFINE:   --ssaf-compilation-unit-id=%{testname}.cu \
// DEFINE:   --ssaf-tu-summary-file=%t/%{testname}.tu.json
// DEFINE: %{link} = clang-ssaf-linker %t/%{testname}.tu.json -o %t/%{testname}.lu.json
// DEFINE: %{analyze} = clang-ssaf-analyzer %t/%{testname}.lu.json \
// DEFINE:   -o %t/%{testname}.wpa.json -a UnsafeBufferReachableAnalysisResult
// DEFINE: %{transform} = mkdir -p %{apply_dir} && %clang -fsyntax-only %t/%{testname}.cpp \
// DEFINE:   --ssaf-source-transformation=cpp-bounded-buffers \
// DEFINE:   --ssaf-global-scope-analysis-result=%t/%{testname}.wpa.json \
// DEFINE:   --ssaf-src-edit-file=%{edits_yaml} \
// DEFINE:   --ssaf-transformation-report-file=%{report_sarif} \
// DEFINE:   --ssaf-compilation-unit-id=%{testname}.cu \
// DEFINE:   --ssaf-link-unit-id=%{testname}.lu
// DEFINE: %{apply} = clang-apply-replacements %{apply_dir}


//--- void_buffer_ptr.cpp
void use(void *p) {
  ((char *)p)[5];
}

//--- void_buffer_ptr.directives
// REDEFINE: %{testname} = void_buffer_ptr
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=VOID_BUFFER_PTR --input-file=%{edits_yaml} %s
// VOID_BUFFER_PTR: Offset: 9
// VOID_BUFFER_PTR: ReplacementText: 'bounded_ptr<char> '
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=VOID_BUFFER_PTR_APPLIED --input-file=%t/%{testname}.cpp %s %}
// VOID_BUFFER_PTR_APPLIED: void use(bounded_ptr<char> p)


//--- pointer_global.cpp
int *p;
void use() { p[5] = 0; }

//--- pointer_global.directives
// REDEFINE: %{testname} = pointer_global
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=POINTER_GLOBAL --input-file=%{edits_yaml} %s
// POINTER_GLOBAL: Offset: 0
// POINTER_GLOBAL: ReplacementText: 'bounded_ptr<int> '
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=POINTER_GLOBAL_APPLIED --input-file=%t/%{testname}.cpp %s %}
// POINTER_GLOBAL_APPLIED: bounded_ptr<int> p;


//--- specifier_static.cpp
static int *p;
void use() { p[5] = 0; }

//--- specifier_static.directives
// REDEFINE: %{testname} = specifier_static
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=SPECIFIER_STATIC --input-file=%{edits_yaml} %s
// SPECIFIER_STATIC: Offset: 7
// SPECIFIER_STATIC: ReplacementText: 'bounded_ptr<int> '
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=SPECIFIER_STATIC_APPLIED --input-file=%t/%{testname}.cpp %s %}
// SPECIFIER_STATIC_APPLIED: static bounded_ptr<int> p;


// ============================================================================
// Qualifiers, and their positions relative to the pointee/element type
// ============================================================================


//--- qualifier_const_leading.cpp
const char *p;
void use() { (void)p[5]; }

//--- qualifier_const_leading.directives
// REDEFINE: %{testname} = qualifier_const_leading
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=QUALIFIER_CONST_LEADING --input-file=%{edits_yaml} %s
// QUALIFIER_CONST_LEADING: Offset: 0
// QUALIFIER_CONST_LEADING: ReplacementText: 'bounded_ptr<const char> '
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=QUALIFIER_CONST_LEADING_APPLIED --input-file=%t/%{testname}.cpp %s %}
// QUALIFIER_CONST_LEADING_APPLIED: bounded_ptr<const char> p;


//--- qualifier_const_trailing_spelled_after.cpp
char const *p;
void use() {
  (void)p[5];
}

//--- qualifier_const_trailing_spelled_after.directives
// REDEFINE: %{testname} = qualifier_const_trailing_spelled_after
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=QUALIFIER_SPELLED_AFTER --input-file=%{edits_yaml} %s
// QUALIFIER_SPELLED_AFTER: Offset: 0
// QUALIFIER_SPELLED_AFTER: ReplacementText: 'bounded_ptr<const char> '
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=QUALIFIER_SPELLED_AFTER_APPLIED --input-file=%t/%{testname}.cpp %s %}
// QUALIFIER_SPELLED_AFTER_APPLIED: bounded_ptr<const char> p;


//--- qualifier_on_pointer_itself.cpp
int *const p = nullptr;
void use() { (void)p[5]; }

//--- qualifier_on_pointer_itself.directives
// REDEFINE: %{testname} = qualifier_on_pointer_itself
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=QUALIFIER_ON_POINTER --input-file=%{edits_yaml} %s
// QUALIFIER_ON_POINTER: Offset: 0
// QUALIFIER_ON_POINTER: ReplacementText: 'bounded_ptr<int> '
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=QUALIFIER_ON_POINTER_APPLIED --input-file=%t/%{testname}.cpp %s %}
// QUALIFIER_ON_POINTER_APPLIED: bounded_ptr<int> const p = nullptr;


//--- qualifier_multiple_trailing.cpp
int *volatile const p = nullptr;
void use() { (void)p[5]; }

//--- qualifier_multiple_trailing.directives
// REDEFINE: %{testname} = qualifier_multiple_trailing
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=QUALIFIER_MULTIPLE_TRAILING --input-file=%{edits_yaml} %s
// QUALIFIER_MULTIPLE_TRAILING: Offset: 0
// QUALIFIER_MULTIPLE_TRAILING: ReplacementText: 'bounded_ptr<int> '
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=QUALIFIER_MULTIPLE_TRAILING_APPLIED --input-file=%t/%{testname}.cpp %s %}
// QUALIFIER_MULTIPLE_TRAILING_APPLIED: bounded_ptr<int> volatile const p = nullptr;


//--- qualifier_array_element_const.cpp
const int arr[3] = {};
void use() { (void)arr[5]; }

//--- qualifier_array_element_const.directives
// REDEFINE: %{testname} = qualifier_array_element_const
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=QUALIFIER_ARRAY_ELEMENT --input-file=%{edits_yaml} %s
// QUALIFIER_ARRAY_ELEMENT: Offset: 0
// QUALIFIER_ARRAY_ELEMENT: ReplacementText: 'bounded_array<const int, 3>'
// QUALIFIER_ARRAY_ELEMENT: Offset: 13
// QUALIFIER_ARRAY_ELEMENT: ReplacementText: ''
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=QUALIFIER_ARRAY_ELEMENT_APPLIED --input-file=%t/%{testname}.cpp %s %}
// QUALIFIER_ARRAY_ELEMENT_APPLIED: bounded_array<const int, 3> arr = {};


//--- qualifier_array_multiple_trailing_reversed.cpp
int volatile const arr[3] = {};
void use() {
  (void)arr[5];
}

//--- qualifier_array_multiple_trailing_reversed.directives
// REDEFINE: %{testname} = qualifier_array_multiple_trailing_reversed
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=QUALIFIER_ARRAY_REVERSED --input-file=%{edits_yaml} %s
// QUALIFIER_ARRAY_REVERSED: Offset: 0
// QUALIFIER_ARRAY_REVERSED: ReplacementText: 'bounded_array<const volatile int, 3>'
// QUALIFIER_ARRAY_REVERSED: Offset: 22
// QUALIFIER_ARRAY_REVERSED: ReplacementText: ''
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=QUALIFIER_ARRAY_REVERSED_APPLIED --input-file=%t/%{testname}.cpp %s %}
// QUALIFIER_ARRAY_REVERSED_APPLIED: bounded_array<const volatile int, 3> arr = {};


//--- qualifier_leading_separated.cpp
const static char *p;
void use() {
  (void)p[5];
}

//--- qualifier_leading_separated.directives
// REDEFINE: %{testname} = qualifier_leading_separated
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=QUALIFIER_LEADING_SEPARATED --input-file=%{edits_yaml} %s
// QUALIFIER_LEADING_SEPARATED: Replacements: []
// RUN: FileCheck --check-prefix=QUALIFIER_LEADING_SEPARATED_REPORT --input-file=%{report_sarif} %s
// QUALIFIER_LEADING_SEPARATED_REPORT: "text": "unexpected token between a leading cv-qualifier and the type"


//--- qualifier_trailing_separated.cpp
int /* c */ const arr[3] = {};
void use() {
  arr[5];
}

//--- qualifier_trailing_separated.directives
// REDEFINE: %{testname} = qualifier_trailing_separated
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=QUALIFIER_TRAILING_SEPARATED --input-file=%{edits_yaml} %s
// QUALIFIER_TRAILING_SEPARATED: Replacements: []
// RUN: FileCheck --check-prefix=QUALIFIER_TRAILING_SEPARATED_REPORT --input-file=%{report_sarif} %s
// QUALIFIER_TRAILING_SEPARATED_REPORT: "text": "unexpected token between the type and a trailing cv-qualifier"


//--- array_plain.cpp
void use() {
  int arr[3];
  arr[5] = 0;
}

//--- array_plain.directives
// REDEFINE: %{testname} = array_plain
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=ARRAY_PLAIN --input-file=%{edits_yaml} %s
// ARRAY_PLAIN: Offset: 15
// ARRAY_PLAIN: ReplacementText: 'bounded_array<int, 3>'
// ARRAY_PLAIN: Offset: 22
// ARRAY_PLAIN: ReplacementText: ''
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=ARRAY_PLAIN_APPLIED --input-file=%t/%{testname}.cpp %s %}
// ARRAY_PLAIN_APPLIED: bounded_array<int, 3> arr;


//--- array_of_pointers.cpp
// An array of pointers: the element type is itself a pointer, but the
// element is not dereferenced by the array rewrite, so it is reproduced
// verbatim inside the angle brackets.
void use() {
  int *arr[3];
  arr[5] = nullptr;
}

//--- array_of_pointers.directives
// REDEFINE: %{testname} = array_of_pointers
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=ARRAY_OF_POINTERS --input-file=%{edits_yaml} %s
// ARRAY_OF_POINTERS: Offset: 198
// ARRAY_OF_POINTERS: ReplacementText: 'bounded_array<int *, 3>'
// ARRAY_OF_POINTERS: Offset: 206
// ARRAY_OF_POINTERS: ReplacementText: ''
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=ARRAY_OF_POINTERS_APPLIED --input-file=%t/%{testname}.cpp %s %}
// ARRAY_OF_POINTERS_APPLIED: bounded_array<int *, 3>arr;


//--- array_multi_dim.cpp
void use() {
  int arr[3][4];
  arr[5][0] = 0;
}

//--- array_multi_dim.directives
// REDEFINE: %{testname} = array_multi_dim
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=ARRAY_MULTI_DIM --input-file=%{edits_yaml} %s
// ARRAY_MULTI_DIM: Replacements: []
// RUN: FileCheck --check-prefix=ARRAY_MULTI_DIM_REPORT --input-file=%{report_sarif} %s
// ARRAY_MULTI_DIM_REPORT: "text": "multi-dimensional array is not yet rewritten"


//--- array_incomplete.cpp
extern int arr[];
void use() {
  arr[5] = 0;
}

//--- array_incomplete.directives
// REDEFINE: %{testname} = array_incomplete
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=ARRAY_INCOMPLETE --input-file=%{edits_yaml} %s
// ARRAY_INCOMPLETE: Replacements: []
// RUN: FileCheck --check-prefix=ARRAY_INCOMPLETE_REPORT --input-file=%{report_sarif} %s
// ARRAY_INCOMPLETE_REPORT: "text": "array of unknown bound is not yet rewritten"


//--- array_of_function_pointers_raw.cpp
void use() {
  void (*arr[4])();
  arr[5]();
}

//--- array_of_function_pointers_raw.directives
// REDEFINE: %{testname} = array_of_function_pointers_raw
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=ARRAY_RAW_FNPTR --input-file=%{edits_yaml} %s
// ARRAY_RAW_FNPTR: Replacements: []
// RUN: FileCheck --check-prefix=ARRAY_RAW_FNPTR_REPORT --input-file=%{report_sarif} %s
// ARRAY_RAW_FNPTR_REPORT: "text": "the array type does not end in a closing bracket"


//--- pointer_local_alias.cpp
void use(int *p) {
  int *q = p;
  q[5] = 0;
}

//--- pointer_local_alias.directives
// REDEFINE: %{testname} = pointer_local_alias
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=POINTER_LOCAL_ALIAS --input-file=%{edits_yaml} %s
// POINTER_LOCAL_ALIAS: Offset: 9
// POINTER_LOCAL_ALIAS: ReplacementText: 'bounded_ptr<int> '
// POINTER_LOCAL_ALIAS: Offset: 21
// POINTER_LOCAL_ALIAS: ReplacementText: 'bounded_ptr<int> '
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=POINTER_LOCAL_ALIAS_APPLIED --input-file=%t/%{testname}.cpp %s %}
// POINTER_LOCAL_ALIAS_APPLIED: void use(bounded_ptr<int> p) {
// POINTER_LOCAL_ALIAS_APPLIED-NEXT: bounded_ptr<int> q = p;


//--- pointer_return_value.cpp
// A reachable function return value.
int *get(int *p) {
  p[5] = 0;
  return p;
}

//--- pointer_return_value.directives
// REDEFINE: %{testname} = pointer_return_value
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=POINTER_RETURN_VALUE --input-file=%{edits_yaml} %s
// POINTER_RETURN_VALUE: Offset: 47
// POINTER_RETURN_VALUE: ReplacementText: 'bounded_ptr<int> '
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=POINTER_RETURN_VALUE_APPLIED --input-file=%t/%{testname}.cpp %s %}
// POINTER_RETURN_VALUE_APPLIED: int *get(bounded_ptr<int> p) {


//--- pointer_multi_level.cpp
void use(int **pp) {
  pp[5] = nullptr;
  (*pp)[5] = 0;
}

//--- pointer_multi_level.directives
// REDEFINE: %{testname} = pointer_multi_level
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=POINTER_MULTI_LEVEL --input-file=%{edits_yaml} %s
// POINTER_MULTI_LEVEL: Replacements: []
// RUN: FileCheck --check-prefix=POINTER_MULTI_LEVEL_REPORT --input-file=%{report_sarif} %s
// POINTER_MULTI_LEVEL_REPORT: "text": "multi-level pointer indirection is not yet rewritten"


//--- pointer_to_array.cpp
void use(int (*p)[3]) {
  p[5][0] = 0;
}

//--- pointer_to_array.directives
// REDEFINE: %{testname} = pointer_to_array
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=POINTER_TO_ARRAY --input-file=%{edits_yaml} %s
// POINTER_TO_ARRAY: Replacements: []
// RUN: FileCheck --check-prefix=POINTER_TO_ARRAY_REPORT --input-file=%{report_sarif} %s
// POINTER_TO_ARRAY_REPORT: "text": "pointer to array is not yet rewritten"


//--- reference_to_pointer.cpp
void use(int *&p) {
  p[5] = 0;
}

//--- reference_to_pointer.directives
// REDEFINE: %{testname} = reference_to_pointer
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=REFERENCE_TO_POINTER --input-file=%{edits_yaml} %s
// REFERENCE_TO_POINTER: Replacements: []
// RUN: FileCheck --check-prefix=REFERENCE_TO_POINTER_REPORT --input-file=%{report_sarif} %s
// REFERENCE_TO_POINTER_REPORT: "text": "reference to pointer is not yet rewritten"


//--- pointer_parenthesized_declarator.cpp
void use() {
  int v = 0;
  int (*p) = &v;
  (void)p[5];
}

//--- pointer_parenthesized_declarator.directives
// REDEFINE: %{testname} = pointer_parenthesized_declarator
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=POINTER_PARENTHESIZED --input-file=%{edits_yaml} %s
// POINTER_PARENTHESIZED: Replacements: []
// RUN: FileCheck --check-prefix=POINTER_PARENTHESIZED_REPORT --input-file=%{report_sarif} %s
// POINTER_PARENTHESIZED_REPORT: "text": "pointer declarator does not end at its '*'"


//--- field_pointer.cpp
struct S{int *p;};
void use(S *w) {
  w->p[5] = 0;
}

//--- field_pointer.directives
// REDEFINE: %{testname} = field_pointer
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=FIELD_POINTER --input-file=%{edits_yaml} %s
// FIELD_POINTER: Offset: 9
// FIELD_POINTER: ReplacementText: 'bounded_ptr<int> '
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=FIELD_POINTER_APPLIED --input-file=%t/%{testname}.cpp %s %}
// FIELD_POINTER_APPLIED: struct S{bounded_ptr<int> p;};


//--- field_array.cpp
// An array-typed struct field.
struct S{int arr[3];};
void use(S *w) {
  w->arr[5] = 0;
}

//--- field_array.directives
// REDEFINE: %{testname} = field_array
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=FIELD_ARRAY --input-file=%{edits_yaml} %s
// FIELD_ARRAY: Offset: 41
// FIELD_ARRAY: ReplacementText: 'bounded_array<int, 3>'
// FIELD_ARRAY: Offset: 48
// FIELD_ARRAY: ReplacementText: ''
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=FIELD_ARRAY_APPLIED --input-file=%t/%{testname}.cpp %s %}
// FIELD_ARRAY_APPLIED: struct S{bounded_array<int, 3> arr;};


// ============================================================================
// Macros
// ============================================================================

//--- macro_expansion.cpp
// Skip MacroExpansion: the declarator's type is spelled through a macro.
#define PTR int *
void use(PTR p) {
  p[5] = 0;
}

//--- macro_expansion.directives
// REDEFINE: %{testname} = macro_expansion
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=MACRO_EXPANSION --input-file=%{edits_yaml} %s
// MACRO_EXPANSION: Replacements: []
// RUN: FileCheck --check-prefix=MACRO_EXPANSION_REPORT --input-file=%{report_sarif} %s
// MACRO_EXPANSION_REPORT: "text": "declarator spelled through a macro is not yet rewritten"


// ============================================================================
// Typedefs
// ============================================================================

//--- typedef_array_of_function_pointers.cpp
// A typedef used as an array element type does not block the array rewrite:
// the typedef keeps the declarator a clean prefix + [N] suffix, so only the
// (unexpanded) element spelling changes.
typedef void (*FP)();
void use() {
  FP arr[4];
  arr[5]();
}

//--- typedef_array_of_function_pointers.directives
// REDEFINE: %{testname} = typedef_array_of_function_pointers
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=TYPEDEF_ARRAY_OF_FNPTRS --input-file=%{edits_yaml} %s
// TYPEDEF_ARRAY_OF_FNPTRS: Offset: 233
// TYPEDEF_ARRAY_OF_FNPTRS: ReplacementText: 'bounded_array<FP, 4>'
// TYPEDEF_ARRAY_OF_FNPTRS: Offset: 239
// TYPEDEF_ARRAY_OF_FNPTRS: ReplacementText: ''
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=TYPEDEF_ARRAY_OF_FNPTRS_APPLIED --input-file=%t/%{testname}.cpp %s %}
// TYPEDEF_ARRAY_OF_FNPTRS_APPLIED: bounded_array<FP, 4> arr;


//--- typedef_pointer.cpp
typedef int *P;
void use(P p) {
  p[5] = 0;
}

//--- typedef_pointer.directives
// REDEFINE: %{testname} = typedef_pointer
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=TYPEDEF_POINTER --input-file=%{edits_yaml} %s
// TYPEDEF_POINTER: Replacements: []
// RUN: FileCheck --check-prefix=TYPEDEF_POINTER_REPORT --input-file=%{report_sarif} %s
// TYPEDEF_POINTER_REPORT: "text": "no TypeLoc for the pointee or array element type"


//--- unnamable_anonymous_struct.cpp
struct { int x; } *p;
void use(int i) {
  p[i].x = 0;
}

//--- unnamable_anonymous_struct.directives
// REDEFINE: %{testname} = unnamable_anonymous_struct
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=UNNAMABLE_ANON_STRUCT --input-file=%{edits_yaml} %s
// UNNAMABLE_ANON_STRUCT: Replacements: []
// RUN: FileCheck --check-prefix=UNNAMABLE_ANON_STRUCT_REPORT --input-file=%{report_sarif} %s
// UNNAMABLE_ANON_STRUCT_REPORT: "text": "the pointee or array element type has no name that can be written as a template argument"


//--- unnamable_lambda_decltype.cpp
void use() {
  auto f = [](int x) { return x; };
  decltype(f) *p = &f;
  (void)p[5];
}

//--- unnamable_lambda_decltype.directives
// REDEFINE: %{testname} = unnamable_lambda_decltype
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=UNNAMABLE_LAMBDA --input-file=%{edits_yaml} %s
// UNNAMABLE_LAMBDA: Replacements: []
// RUN: FileCheck --check-prefix=UNNAMABLE_LAMBDA_REPORT --input-file=%{report_sarif} %s
// UNNAMABLE_LAMBDA_REPORT: "text": "the pointee or array element type has no name that can be written as a template argument"


//--- whitespace_free_pointer.cpp
int*p;
void use(){p[5]=0;}

//--- whitespace_free_pointer.directives
// REDEFINE: %{testname} = whitespace_free_pointer
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=WHITESPACE_FREE_POINTER --input-file=%{edits_yaml} %s
// WHITESPACE_FREE_POINTER: Offset: 0
// WHITESPACE_FREE_POINTER: ReplacementText: 'bounded_ptr<int> '
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=WHITESPACE_FREE_POINTER_APPLIED --input-file=%t/%{testname}.cpp %s %}
// WHITESPACE_FREE_POINTER_APPLIED: bounded_ptr<int> p;


//--- skip_declaration_group.cpp
void use() {
  int *a, *b;
  a[5] = 0;
  b[5] = 0;
}

//--- skip_declaration_group.directives
// REDEFINE: %{testname} = skip_declaration_group
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=SKIP_DECL_GROUP --input-file=%{edits_yaml} %s
// SKIP_DECL_GROUP: Replacements: []
// RUN: FileCheck --check-prefix=SKIP_DECL_GROUP_REPORT --input-file=%{report_sarif} %s
// SKIP_DECL_GROUP_REPORT-DAG: "text": "declarator of a multi-declarator group is not yet rewritten"


//--- skip_trailing_return_type.cpp
auto f(int *p) -> int * {
  return p;
}
void use() {
  int *q = f(nullptr);
  q[5] = 0;
}

//--- skip_trailing_return_type.directives
// REDEFINE: %{testname} = skip_trailing_return_type
// RUN: %{extract}
// RUN: %{link}
// RUN: %{analyze}
// RUN: %{transform}
// RUN: FileCheck --check-prefix=SKIP_TRAILING_RETURN_REPORT --input-file=%{report_sarif} %s
// SKIP_TRAILING_RETURN_REPORT: "text": "trailing return type is not yet rewritten"
// RUN: FileCheck --check-prefix=SKIP_TRAILING_RETURN --input-file=%{edits_yaml} %s
// SKIP_TRAILING_RETURN: Offset: 7
// SKIP_TRAILING_RETURN: ReplacementText: 'bounded_ptr<int> '
// SKIP_TRAILING_RETURN: Offset: 55
// SKIP_TRAILING_RETURN: ReplacementText: 'bounded_ptr<int> '
// RUN: %if clang-apply-replacements %{ %{apply} %}
// RUN: %if clang-apply-replacements %{ FileCheck --check-prefix=SKIP_TRAILING_RETURN_APPLIED --input-file=%t/%{testname}.cpp %s %}
// SKIP_TRAILING_RETURN_APPLIED: auto f(bounded_ptr<int> p) -> int * {
// SKIP_TRAILING_RETURN_APPLIED: bounded_ptr<int> q = f(nullptr);
