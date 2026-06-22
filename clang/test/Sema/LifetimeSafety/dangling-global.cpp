// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety -Wno-dangling -verify %s
#include "Inputs/lifetime-analysis.h"

int *global; // expected-note 10 {{this global dangles}}
int *global_backup; // expected-note {{this global dangles}}

std::string_view global_view; // expected-note {{this global dangles}}
void takeString(std::string&& s);

struct ObjWithStaticField {
  static int *static_field; // expected-note {{this static storage dangles}}
}; 

void save_global() {
  global_backup = global;
}

// Here, by action of save_global, we have that global_backup points to stack memory. This is currently not caught.
void invoke_function_with_side_effects() {
  int local;
  global = &local;
  save_global(); 
  global = nullptr;
} 

// We can however catch the inlined one of course!
void inlined() {
  int local;
  global = &local; // expected-warning {{stack memory associated with local variable 'local' escapes to the global variable 'global_backup' which will dangle}}
  global_backup = global; 
  global = nullptr;
}

void store_local_in_global() {
  int local;
  global = &local; // expected-warning {{stack memory associated with local variable 'local' escapes to the global variable 'global' which will dangle}}
}

int side();
void store_local_in_global_via_comma() {
  int local;
  global = (side(), &local); // expected-warning {{stack memory associated with local variable 'local' escapes to the global variable 'global' which will dangle}}
}

void store_then_clear() {
  int local;
  global = &local;
  global = nullptr;
}

void dangling_static_field() {
  int local;
  ObjWithStaticField::static_field = &local; // expected-warning {{stack memory associated with local variable 'local' escapes to the static variable 'static_field' which will dangle}}
}

// A store on some-but-not-all paths must still be caught: the global's origin
// only spans blocks via the function-exit escape, so it must survive the join.
void conditional_escape(int c) {
  int local = 7;
  if (c)
    global = &local; // expected-warning {{stack memory associated with local variable 'local' escapes to the global variable 'global' which will dangle}}
}

void loop_escape(int n) {
  int local = 0;
  for (int i = 0; i < n; ++i)
    global = &local; // expected-warning {{stack memory associated with local variable 'local' escapes to the global variable 'global' which will dangle}}
}

// Negative: a conditional store that never leaks a stack address is silent.
void conditional_no_escape(int c) {
  int local = 7;
  if (c)
    global = nullptr; // no-warning
  (void)local;
}

// Pointer compound assignment and increment/decrement keep the pointer in the
// same allocation, so the result carries the borrow.
void via_compound_add() {
  int local[10];
  int *p = local; // expected-warning {{stack memory associated with local variable 'local' escapes to the global variable 'global' which will dangle}}
  global = (p += 1);
}

void via_compound_sub() {
  int local[10];
  int *p = local + 5; // expected-warning {{stack memory associated with local variable 'local' escapes to the global variable 'global' which will dangle}}
  global = (p -= 1);
}

void via_preinc() {
  int local[10];
  int *p = local; // expected-warning {{stack memory associated with local variable 'local' escapes to the global variable 'global' which will dangle}}
  global = ++p;
}

void via_postinc() {
  int local[10];
  int *p = local; // expected-warning {{stack memory associated with local variable 'local' escapes to the global variable 'global' which will dangle}}
  global = p++;
}

void via_predec() {
  int local[10];
  int *p = local + 5; // expected-warning {{stack memory associated with local variable 'local' escapes to the global variable 'global' which will dangle}}
  global = --p;
}

void via_postdec() {
  int local[10];
  int *p = local + 5; // expected-warning {{stack memory associated with local variable 'local' escapes to the global variable 'global' which will dangle}}
  global = p--;
}

// Negative: arithmetic on a pointer into long-lived storage stays silent.
void ok_global_storage() {
  static int s[10];
  int *p = s;
  p += 1;
  ++p;
  global = (p -= 1); // no-warning
}
// When a local string is stored in a global view and then moved, the analyzer warns it "may" dangle since the storage may have been moved
void store_local_in_global_but_moved(std::string s){
  global_view = s; // expected-warning-re {{stack memory associated with parameter 's' may escape to the global variable 'global_view' which will dangle{{.*}} may have been moved}}
  takeString(std::move(s)); //expected-note {{potentially moved here}}
}
