// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -Wuninitialized \
// RUN:   -debug-info-kind=line-tables-only -emit-obj -o /dev/null -verify=warn %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -Wall \
// RUN:   -debug-info-kind=line-tables-only -emit-obj -o /dev/null -verify=warn %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 \
// RUN:   -Wconditional-uninitialized -debug-info-kind=line-tables-only \
// RUN:   -emit-obj -o /dev/null -verify=maybe %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -Wuninitialized \
// RUN:   -Wconditional-uninitialized -debug-info-kind=line-tables-only \
// RUN:   -emit-obj -o /dev/null -verify=warn,maybe %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -Wuninitialized \
// RUN:   -Wconditional-uninitialized -emit-obj -o /dev/null \
// RUN:   -verify=imprecise %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -Wuninitialized \
// RUN:   -emit-obj -o /dev/null -verify=disabled %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -Wno-uninitialized \
// RUN:   -emit-obj -o /dev/null -verify=disabled %s
// disabled-no-diagnostics

void consume(int);

struct C {
  int i;
  C() {}
  void set(int value) { i = value; }
  void use() {
    // warn-warning@+2 4 {{field is uninitialized when used here}}
    // maybe-warning@+1 2 {{field may be uninitialized when used here}}
    consume(i);
  }
};

void unknown(C &);

void stack_uninitialized() { // imprecise-warning {{field is uninitialized when used here}}
  C c;
  c.use();
}

void stack_initialized() {
  C c;
  c.set(1);
  c.use();
}

void stack_conditionally_initialized(bool condition) { // imprecise-warning {{field may be uninitialized when used here}}
  C c;
  if (condition)
    c.set(1);
  c.use();
}

void heap_uninitialized() { // imprecise-warning {{field is uninitialized when used here}}
  C *c = new C;
  c->use();
}

void heap_initialized() {
  C *c = new C;
  c->set(1);
  c->use();
}

void heap_conditionally_initialized(bool condition) { // imprecise-warning {{field may be uninitialized when used here}}
  C *c = new C;
  if (condition)
    c->set(1);
  c->use();
}

void heap_unknown_call() {
  C *c = new C;
  unknown(*c);
  c->use();
}

struct Owner {
  C *pointer;
  C *operator->() { return pointer; }
};

void owner_uninitialized() { // imprecise-warning {{field is uninitialized when used here}}
  Owner owner{new C};
  owner->use();
}

struct Box {
  long control[2];
  C object;
};

void nonzero_offset_uninitialized() { // imprecise-warning {{field is uninitialized when used here}}
  Box *box = new Box;
  box->object.use();
}

enum class Cache : unsigned char { No, Yes, Unknown };

struct BitFields {
  unsigned precedence : 6;
  Cache rhs : 2;
  Cache array : 2;
  Cache function : 2;

  BitFields(unsigned p, Cache r, Cache a, Cache f)
      : precedence(p), rhs(r), array(a), function(f) {}
};

BitFields *initialize_bit_fields(unsigned p, Cache r, Cache a, Cache f) {
  return new BitFields(p, r, a, f);
}
