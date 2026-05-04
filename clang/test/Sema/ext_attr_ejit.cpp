// RUN: %clang_cc1 -fsyntax-only -verify=expected %s
// EmbeddedJIT attribute semantic analysis tests

// === Correct usage -- should produce no diagnostics ===

struct CellConfig {
  __attribute__((ejit_may_const)) int cellType;
  __attribute__((ejit_may_const)) unsigned flags;
  __attribute__((ejit_may_const)) float ratio;
  int xx;
};

__attribute__((ejit_period("static"))) int g_board;
__attribute__((ejit_period_arr("cell"))) struct CellConfig g_cells[16];

__attribute__((ejit_entry))
void process_task(__attribute__((ejit_period_arr_ind("cell"))) int idx);

__attribute__((ejit_period_lc("cell")))
void update_config(__attribute__((ejit_period_arr_ind("cell"))) int idx);

// === Error: ejit_period on array ===
__attribute__((ejit_period("cell"))) int g_bad_array[10];
// expected-error@-1 {{ejit_period attribute cannot be used on array variable 'g_bad_array'; use ejit_period_arr for arrays}}

// === Error: ejit_period_arr on non-array ===
__attribute__((ejit_period_arr("cell"))) int g_not_array;
// expected-error@-1 {{ejit_period_arr attribute requires an array type; 'g_not_array' is not an array}}

// === Error: duplicate period attributes (only second usage triggers error) ===
__attribute__((ejit_period("one")))
__attribute__((ejit_period("two")))
// expected-error@-1 {{variable 'g_conflict' cannot have multiple ejit_period or ejit_period_arr attributes}}
int g_conflict;

// === Error: ejit_period_arr_ind on non-integer parameter ===
__attribute__((ejit_entry))
void bad_ind_type(__attribute__((ejit_period_arr_ind("cell"))) float badIdx);
// expected-error@-1 {{ejit_period_arr_ind parameter 'badIdx' must have integer type}}

// === Error: too many ejit_period_arr_ind parameters ===
__attribute__((ejit_entry))
void too_many_ind(
    __attribute__((ejit_period_arr_ind("a"))) int a,
    __attribute__((ejit_period_arr_ind("b"))) int b,
    __attribute__((ejit_period_arr_ind("c"))) int c,
    __attribute__((ejit_period_arr_ind("d"))) int d,
    __attribute__((ejit_period_arr_ind("e"))) int e);
// expected-error@-1 {{function 'too_many_ind' has 5 ejit_period_arr_ind parameters, which exceeds the maximum of 4}}

// === Error: ejit_period_lc without matching ind parameter ===
// Note: %0 is printed without quotes for string arguments
__attribute__((ejit_period_lc("nonexistent")))
// expected-error@-1 {{ejit_period_lc(nonexistent) requires a corresponding ejit_period_arr_ind(nonexistent) parameter}}
void bad_lc(int x);
