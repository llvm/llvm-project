// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -Wuninitialized -debug-info-kind=line-tables-only -emit-obj -o /dev/null -verify=warn,late %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -Wall -debug-info-kind=line-tables-only -emit-obj -o /dev/null -verify=warn,late %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -Wuninitialized -debug-info-kind=line-tables-only -emit-obj -o /dev/null -verify=warn %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -Wconditional-uninitialized -debug-info-kind=line-tables-only -emit-obj -o /dev/null -verify=maybe %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -Wconditional-uninitialized -debug-info-kind=line-tables-only -emit-obj -o /dev/null -verify=maybe %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -Wuninitialized -Wconditional-uninitialized -debug-info-kind=line-tables-only -emit-obj -o /dev/null -verify=warn,late,maybe %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -Wuninitialized -Wconditional-uninitialized -emit-obj -o /dev/null -verify=imprecise %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -Wuninitialized -emit-llvm -o - %s 2>/dev/null | FileCheck %s --check-prefix=NO-DEBUG-IR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -Wno-uninitialized -emit-obj -o /dev/null -verify=disabled %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -emit-obj -o /dev/null -verify=disabled %s
// disabled-no-diagnostics
// NO-DEBUG-IR: define
// NO-DEBUG-IR-NOT: !dbg

void consume(int);

struct C {
  int i;
  int j;
  C() {}
  void set_i() { i = 1; }
  void set_j() { j = 1; }
  void inspect() const { consume(j); } // late-warning {{field is uninitialized when used here}}
};

// warn-warning@+1 {{field is uninitialized when used here}}
void no_write() { C c; consume(c.i); } // imprecise-warning {{field is uninitialized when used here}}

void same_field_write() { C c; c.set_i(); consume(c.i); }

// warn-warning@+1 {{field is uninitialized when used here}}
void sibling_field_write() { C c; c.set_j(); consume(c.i); } // imprecise-warning {{field is uninitialized when used here}}

void unknown(C &);
void unknown_call() { C c; unknown(c); consume(c.i); }

// warn-warning@+1 {{field is uninitialized when used here}}
void readonly_call() { C c; c.inspect(); consume(c.i); } // imprecise-warning 2 {{field is uninitialized when used here}}

void conditional_write(bool condition) { // imprecise-warning {{field may be uninitialized when used here}}
  C c;
  if (condition)
    c.i = 1;
  // maybe-warning@+1 {{field may be uninitialized when used here}}
  consume(c.i);
}

void conditional_sibling_write(bool condition) { // imprecise-warning {{field is uninitialized when used here}}
  C c;
  if (condition)
    c.j = 1;
  // warn-warning@+1 {{field is uninitialized when used here}}
  consume(c.i);
}

struct B {
  int i;
  int j;
};

struct F {
  int padding;
  B b;
};

// warn-warning@+1 {{field is uninitialized when used here}}
void copy_uninitialized() { B b; F f; f.b = b; consume(f.b.i); } // imprecise-warning {{field is uninitialized when used here}}

void copy_initialized() { B b; b.i = 1; F f; f.b = b; consume(f.b.i); }

// warn-warning@+1 {{field is uninitialized when used here}}
void copy_sibling_initialized() { B b; b.j = 1; F f; f.b = b; consume(f.b.i); } // imprecise-warning {{field is uninitialized when used here}}

C *escaped;
void escaped_address() { C c; escaped = &c; consume(c.i); }
