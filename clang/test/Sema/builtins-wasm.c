// RUN: %clang_cc1 -fsyntax-only -verify -triple wasm32 -target-feature +reference-types %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple wasm32 -target-abi experimental-mv -DMULTIVALUE -target-feature +reference-types %s

#define EXPR_HAS_TYPE(expr, type) _Generic((expr), type : 1, default : 0)

static __externref_t table[0];

typedef void (*__funcref funcref_t)();
void test_ref_null() {
  funcref_t func = __builtin_wasm_ref_null_func(0); // expected-error {{too many arguments to function call, expected 0, have 1}}
  __externref_t ref = __builtin_wasm_ref_null_extern(0); // expected-error {{too many arguments to function call, expected 0, have 1}}
  __builtin_wasm_ref_is_null_extern(ref, 1); // expected-error {{too many arguments to function call, expected 1, have 2}}
  __builtin_wasm_ref_is_null_extern(); // expected-error {{too few arguments to function call, expected 1, have 0}}
  __builtin_wasm_ref_is_null_extern(1); // expected-error {{1st argument must be an externref}}
}

void test_table_size(__externref_t ref, void *ptr, int arr[]) {
  __builtin_wasm_table_size();                        // expected-error {{too few arguments to function call, expected 1, have 0}}
  __builtin_wasm_table_size(1);                       // expected-error {{1st argument must be a WebAssembly table}}
  __builtin_wasm_table_size(ref);                     // expected-error {{1st argument must be a WebAssembly table}}
  __builtin_wasm_table_size(ptr);                     // expected-error {{1st argument must be a WebAssembly table}}
  __builtin_wasm_table_size(arr);                     // expected-error {{1st argument must be a WebAssembly table}}
  __builtin_wasm_table_size(table, table);            // expected-error {{too many arguments to function call, expected 1, have 2}}

  _Static_assert(EXPR_HAS_TYPE(__builtin_wasm_table_size(table), unsigned long), "");
}

void test_table_grow(__externref_t ref, int size) {
  __builtin_wasm_table_grow();                           // expected-error {{too few arguments to function call, expected 3, have 0}}
  __builtin_wasm_table_grow(table, table, table, table); // expected-error {{too many arguments to function call, expected 3, have 4}}
  __builtin_wasm_table_grow(ref, ref, size);             // expected-error {{1st argument must be a WebAssembly table}}
  __builtin_wasm_table_grow(table, table, size);         // expected-error {{2nd argument must match the element type of the WebAssembly table in the 1st argument}}
  __builtin_wasm_table_grow(table, ref, table);          // expected-error {{3rd argument must be an integer}}

  _Static_assert(EXPR_HAS_TYPE(__builtin_wasm_table_grow(table, ref, size), int), "");
}

void test_table_fill(int index, __externref_t ref, int nelem) {
  __builtin_wasm_table_fill();                                   // expected-error {{too few arguments to function call, expected 4, have 0}}
  __builtin_wasm_table_fill(table, table, table, table, table);  // expected-error {{too many arguments to function call, expected 4, have 5}}
  __builtin_wasm_table_fill(index, index, ref, nelem);           // expected-error {{1st argument must be a WebAssembly table}}
  __builtin_wasm_table_fill(table, table, ref, nelem);           // expected-error {{2nd argument must be an integer}}
  __builtin_wasm_table_fill(table, index, index, ref);           // expected-error {{3rd argument must match the element type of the WebAssembly table in the 1st argument}}
  __builtin_wasm_table_fill(table, index, ref, table);           // expected-error {{4th argument must be an integer}}
  __builtin_wasm_table_fill(table, index, ref, nelem);
}

void test_table_copy(int dst_idx, int src_idx, int nelem) {
  __builtin_wasm_table_copy();                                         // expected-error {{too few arguments to function call, expected 5, have 0}}
  __builtin_wasm_table_copy(table, table, table, table, table, table); // expected-error {{too many arguments to function call, expected 5, have 6}}
  __builtin_wasm_table_copy(src_idx, table, dst_idx, src_idx, nelem);  // expected-error {{1st argument must be a WebAssembly table}}
  __builtin_wasm_table_copy(table, src_idx, dst_idx, src_idx, nelem);  // expected-error {{2nd argument must be a WebAssembly table}}
  __builtin_wasm_table_copy(table, table, table, src_idx, nelem);      // expected-error {{3rd argument must be an integer}}
  __builtin_wasm_table_copy(table, table, dst_idx, table, nelem);      // expected-error {{4th argument must be an integer}}
  __builtin_wasm_table_copy(table, table, dst_idx, src_idx, table);    // expected-error {{5th argument must be an integer}}
  __builtin_wasm_table_copy(table, table, dst_idx, src_idx, nelem);
}

typedef void (*F1)(void);
typedef int (*F2)(int);
typedef void (*F3)(struct {int x; double y;});
typedef struct {int x; double y;} (*F4)(void);

void test_function_pointer_signature() {
  // Test argument count validation
  (void)__builtin_wasm_test_function_pointer_signature(); // expected-error {{too few arguments to function call, expected 1, have 0}}
  (void)__builtin_wasm_test_function_pointer_signature((F1)0, (F2)0); // expected-error {{too many arguments to function call, expected 1, have 2}}

  // // Test argument type validation - should require function pointer
  (void)__builtin_wasm_test_function_pointer_signature((void*)0); // expected-error {{used type 'void *' where function pointer is required}}
  (void)__builtin_wasm_test_function_pointer_signature((int)0);   // expected-error {{used type 'int' where function pointer is required}}

  // // Test valid usage
  int res = __builtin_wasm_test_function_pointer_signature((F1)0);
  res = __builtin_wasm_test_function_pointer_signature((F2)0);

  // Test return type
  _Static_assert(EXPR_HAS_TYPE(__builtin_wasm_test_function_pointer_signature((F1)0), int), "");

#ifdef MULTIVALUE
  // Test that struct arguments and returns are rejected with multivalue abi
  (void)__builtin_wasm_test_function_pointer_signature((F3)0); // expected-error {{not supported with the multivalue ABI for function pointers with a struct/union as parameter}}
  (void)__builtin_wasm_test_function_pointer_signature((F4)0); // expected-error {{not supported with the multivalue ABI for function pointers with a struct/union as return value}}
#else
  // with default abi they are fine
  (void)__builtin_wasm_test_function_pointer_signature((F3)0);
  (void)__builtin_wasm_test_function_pointer_signature((F4)0);
#endif
}
