// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.cplusplus.DanglingPtrDeref \
// RUN:   -analyzer-config cfg-lifetime=true -analyzer-output=text -verify %s

void test_case_one() {
  int *ptr = nullptr;
  {
    int num = 5; // expected-note {{'num' initialized to 5}}
    ptr = &num; // expected-note  {{Value assigned to 'ptr'}}
  }
  // expected-note@-1 {{'num' is destroyed here}}
  *ptr = 6;
  // expected-warning@-1 {{Use of 'num' after its lifetime ended}}
  // expected-note@-2    {{Use of 'num' after its lifetime ended}}
}

void test_case_two() {
  int *ptr_one = nullptr;
  int *ptr_two = nullptr;
  {
    int n = 1; // expected-note {{'n' initialized to 1}}
    int m = 2; // expected-note {{'m' initialized to 2}}
    ptr_one = &n; // expected-note {{Value assigned to 'ptr_one'}}
    ptr_two = &m; // expected-note {{Value assigned to 'ptr_two'}}
  }
  // expected-note@-1 {{'n' is destroyed here}}
  // expected-note@-2 {{'m' is destroyed here}}
  *ptr_one = 6;
  *ptr_two = 7; 
  // expected-warning@-2 {{Use of 'n' after its lifetime ended}}
  // expected-note@-3    {{Use of 'n' after its lifetime ended}}
  // expected-warning@-3 {{Use of 'm' after its lifetime ended}}
  // expected-note@-4    {{Use of 'm' after its lifetime ended}}
}

void escape(int *ptr);

void test_case_three() {
  int num = 5;
  int *ptr = &num;
  {
    *ptr = 6; // no-warning
  }
}

void test_case_four() {
  int *ptr = nullptr;
  {
    int num = 5; // expected-note {{'num' initialized to 5}}
    ptr = &num;
  }
  // expected-note@-1 {{'num' is destroyed here}}
  int i = *ptr;
  // expected-warning@-1 {{Use of 'num' after its lifetime ended}}
  // expected-note@-2    {{Use of 'num' after its lifetime ended}}
  i += i;
}

void test_case_five() {
  int *ptr = nullptr;
  for(int i = 0; i < 10; ++i) {
    ptr = &i;
  }
  escape(ptr); // no-warning
}

void test_case_six() {
  for(int i = 0; i < 10; ++i) {
    int *ptr = &i;
    escape(ptr); // no-warning
  }
}

void test_case_seven() {
  int *ptr = nullptr;
  // expected-note@+3 {{Loop condition is true.  Entering loop body}}
  // expected-note@+2 {{Assuming 'i' is >= 10}}
  // expected-note@+1 {{Loop condition is false. Execution continues on line}}
  for (int i = 0; i < 10; ++i) { // expected-note {{'i' initialized to 0}}
    ptr = &i; // expected-note {{Value assigned to 'ptr'}}
    escape(ptr);
  }
  // expected-note@-1 {{'i' is destroyed here}}
  *ptr = 6;
  // expected-warning@-1 {{Use of 'i' after its lifetime ended}}
  // expected-note@-2    {{Use of 'i' after its lifetime ended}}
}

void passing_dangling_ptr_to_opaque_func() {
  int *ptr = nullptr;
  {
    int num = 5; // expected-note {{'num' initialized to 5}}
    ptr = &num; // expected-note  {{Value assigned to 'ptr'}}
  }
  // expected-note@-1 {{'num' is destroyed here}}
  escape(ptr);
  // expected-warning@-1 {{Use of 'num' after its lifetime ended}}
  // expected-note@-2    {{Use of 'num' after its lifetime ended}}
}

int deref_param(int *p) { return *p; }
// expected-warning@-1 {{Use of 'num' after its lifetime ended}}
// expected-note@-2    {{Use of 'num' after its lifetime ended}}

void inlined_callee_single_report() {
  int *ptr = nullptr;
  {
    int num = 5; // expected-note {{'num' initialized to 5}}
    ptr = &num;
  }
  // expected-note@-1 {{'num' is destroyed here}}
  int r = deref_param(ptr);
  // expected-note@-1 {{Calling 'deref_param'}}
  (void)r;
}

struct MyBuffer {
  char buffer[8];
};
struct MyStruct { int x; };

struct Inner { int x; };

struct Outer { struct Inner inner; };

struct A {
  struct B { int x; };
  B b;
};


char member_subregion_dangling_deref() {
  const char *p = nullptr;
  {
    MyBuffer tmp_buffer = {}; // expected-note {{Initializing to 0}}
    p = tmp_buffer.buffer;
  }
  // expected-note@-1 {{'tmp_buffer.buffer[0]' is destroyed here}}
  return *p; 
  // expected-warning@-1 {{Use of 'tmp_buffer.buffer[0]' after its lifetime ended}}
  // expected-note@-2    {{Use of 'tmp_buffer.buffer[0]' after its lifetime ended}}
}

void opaque(const char *);
void opaque_pp(const char **);

void passing_dangling_to_call() {
  const char *p = nullptr;
  {
    MyBuffer tmp_buffer = {};
    p = tmp_buffer.buffer; // expected-note {{Value assigned to 'p'}}
  }
  // expected-note@-1 {{'tmp_buffer.buffer[0]' is destroyed here}}
  opaque(p);
  // expected-warning@-1 {{Use of 'tmp_buffer.buffer[0]' after its lifetime ended}}
  // expected-note@-2    {{Use of 'tmp_buffer.buffer[0]' after its lifetime ended}}
}

char member_subregion_alive_deref() {
  {
    MyBuffer tmp_buffer = {};
    const char *p = tmp_buffer.buffer;
    opaque(p); // no-warning
    return *p; // no-warning
  }
}

char member_subregion_alive_deref_pp() {
  const char *ptr = nullptr;
  const char **pp = &ptr;
  {
    MyBuffer tmp_buffer = {};
    ptr = tmp_buffer.buffer;
  }
  opaque_pp(pp);
  return **pp; // no-warning  
}

void arr_elem_subreg_dangling_deref() {
  int *ptr = nullptr;
  {
    int local_arr[4]; // expected-note    {{'local_arr' declared without an initial value}}
    ptr = &local_arr[1]; // expected-note {{Value assigned to 'ptr'}}
  }
  // expected-note@-1 {{'local_arr[1]' is destroyed here}}
  *ptr = 7;
  // expected-warning@-1 {{Use of 'local_arr[1]' after its lifetime ended}}
  // expected-note@-2    {{Use of 'local_arr[1]' after its lifetime ended}}
}

char member_array_elem_dangling_deref() {
  const char *p = nullptr;
  {
    MyBuffer tmp_buffer = {}; // expected-note {{Initializing to 0}}
    p = tmp_buffer.buffer + 3;
  }
  // expected-note@-1 {{'tmp_buffer.buffer[3]' is destroyed here}}
  return *p;
  // expected-warning@-1 {{Use of 'tmp_buffer.buffer[3]' after its lifetime ended}}
  // expected-note@-2    {{Use of 'tmp_buffer.buffer[3]' after its lifetime ended}}
}

char member_array_out_of_bounds_dangling_deref() {
  const char *p = nullptr;
  {
    MyBuffer tmp_buffer = {}; // expected-note {{Initializing to 0}}
    p = tmp_buffer.buffer + 10;
  }
  // expected-note@-1 {{'tmp_buffer.buffer[10]' is destroyed here}}
  return *p;
  // expected-warning@-1 {{Use of 'tmp_buffer.buffer[10]' after its lifetime ended}}
  // expected-note@-2    {{Use of 'tmp_buffer.buffer[10]' after its lifetime ended}}
}

int struct_field_dangling_deref() {
  int *p = nullptr;
  {
    MyStruct s = {}; // expected-note {{'s.x' initialized to 0}}
    p = &s.x;
  }
  // expected-note@-1 {{'s.x' is destroyed here}}
  return *p;
  // expected-warning@-1 {{Use of 's.x' after its lifetime ended}}
  // expected-note@-2    {{Use of 's.x' after its lifetime ended}}
}

int struct_array_element_dangling_deref() {
  int *p = nullptr;
  {
    MyStruct arr[4] = {}; // expected-note {{field 'x' initialized to 0}}
    p = &arr[2].x;
  }
  // expected-note@-1 {{'arr[2].x' is destroyed here}}
  return *p;
  // expected-warning@-1 {{Use of 'arr[2].x' after its lifetime ended}}
  // expected-note@-2    {{Use of 'arr[2].x' after its lifetime ended}}
}

int nested_field_dangling_deref() {
  int *p = nullptr;
  {
    Outer o = {}; // expected-note {{'o.inner.x' initialized to 0}}
    p = &o.inner.x;
  }
  // expected-note@-1 {{'o.inner.x' is destroyed here}}
  return *p;
  // expected-warning@-1 {{Use of 'o.inner.x' after its lifetime ended}}
  // expected-note@-2    {{Use of 'o.inner.x' after its lifetime ended}}
}

int nested_type_field_dangling_deref() {
  int *p = nullptr;
  {
    A a = {}; // expected-note {{'a.b.x' initialized to 0}}
    p = &a.b.x;
  }
  // expected-note@-1 {{'a.b.x' is destroyed here}}
  return *p;
  // expected-warning@-1 {{Use of 'a.b.x' after its lifetime ended}}
  // expected-note@-2    {{Use of 'a.b.x' after its lifetime ended}}
}

char member_subregion_dangling_deref_increment() {
  const char *p = nullptr;
  {
    MyBuffer tmp_buffer = {}; // expected-note {{Initializing to 0}}
    p = tmp_buffer.buffer;
  }
  // expected-note@-1 {{'tmp_buffer.buffer[1]' is destroyed here}}
  p++;
  return *p;
  // expected-warning@-1 {{Use of 'tmp_buffer.buffer[1]' after its lifetime ended}}
  // expected-note@-2    {{Use of 'tmp_buffer.buffer[1]' after its lifetime ended}}
}
