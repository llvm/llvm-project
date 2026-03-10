// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety -Wno-dangling -verify %s

int *global; // expected-note {{this global dangles}}
int *global_backup; // expected-note {{this global dangles}}

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
  global = &local; // expected-warning {{address of stack memory escapes to global or static storage}}
  global_backup = global; 
  global = nullptr;
}

void store_local_in_global() {
  int local;
  global = &local; // expected-warning {{address of stack memory escapes to global or static storage}}
}

void store_then_clear() {
  int local;
  global = &local;
  global = nullptr;
}

void dangling_static_field() {
  int local;
  ObjWithStaticField::static_field = &local; // expected-warning {{address of stack memory escapes to global or static storage}}
}