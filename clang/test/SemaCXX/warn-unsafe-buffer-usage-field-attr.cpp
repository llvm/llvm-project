// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions -verify %s

using size_t = __typeof(sizeof(int));

namespace std {
  class type_info;
  class bad_cast;
  class bad_typeid;

  template <typename T> class span {

  private:
    T *elements;
    size_t size_;

  public:
    span(T *, size_t){}

    constexpr T* data() const noexcept {
      return elements;
    }

    constexpr size_t size() const noexcept {
      return size_;
    }

  };
}

struct A {
    [[clang::unsafe_buffer_usage]]
    int *ptr;

    size_t sz;
};

struct B {
   A a;
 
   [[clang::unsafe_buffer_usage]]
   int buf[];
};

struct D { 
  [[clang::unsafe_buffer_usage]]
  int *ptr, *ptr2;

  [[clang::unsafe_buffer_usage]]
  int buf[10];
 
  size_t sz;
  
};

void foo(int *ptr);

void foo_safe(std::span<int> sp);

int* test_atribute_struct(A a) {
   int b = *(a.ptr); //expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
   a.sz++;
   // expected-warning@+1{{unsafe pointer arithmetic}}
   return a.ptr++; //expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
}

void test_attribute_field_deref_chain(B b) {
  int *ptr = b.a.ptr;//expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
  foo(b.buf); //expected-warning{{field 'buf' prone to unsafe buffer manipulation}}
}

void test_writes_from_span(std::span<int> sp) {
  A a;
  a.ptr = sp.data(); //expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
  a.sz = sp.size();

  a.ptr = nullptr; // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
}

void test_reads_to_span(A a, A b) {
  //expected-warning@+1{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
  std::span<int> sp {a.ptr, a.sz}; //expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}

  // expected-warning@+1 3{{field 'ptr' prone to unsafe buffer manipulation}}
  if(a.ptr != nullptr && a.ptr != b.ptr) {
    foo_safe(sp);
  }

}

void test_attribute_multiple_fields (D d) {
   int *p =d.ptr; //expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
   p = d.ptr2; //expected-warning{{field 'ptr2' prone to unsafe buffer manipulation}}

   p = d.buf; //expected-warning{{field 'buf' prone to unsafe buffer manipulation}}

   int v = d.buf[0]; //expected-warning{{field 'buf' prone to unsafe buffer manipulation}}

   v = d.buf[5]; //expected-warning{{field 'buf' prone to unsafe buffer manipulation}}
}

template <typename T>
struct TemplateArray {
  [[clang::unsafe_buffer_usage]]
  T *buf;

  [[clang::unsafe_buffer_usage]]
  size_t sz;
};


void test_struct_template (TemplateArray<int> t) {
  int *p = t.buf; //expected-warning{{field 'buf' prone to unsafe buffer manipulation}}
  size_t s = t.sz; //expected-warning{{field 'sz' prone to unsafe buffer manipulation}}
}

class R {
  [[clang::unsafe_buffer_usage]]
  int *array;

  public:
   int* getArray() {
     return array; //expected-warning{{field 'array' prone to unsafe buffer manipulation}}
   }
 
   void setArray(int *arr) {
     array = arr; //expected-warning{{field 'array' prone to unsafe buffer manipulation}}
   }
};

template<class P>
class Q {
  [[clang::unsafe_buffer_usage]]
  P *array;

  public:
   P* getArray() {
     return array; //expected-warning{{field 'array' prone to unsafe buffer manipulation}}
   }

   void setArray(P *arr) {
     array = arr; //expected-warning{{field 'array' prone to unsafe buffer manipulation}}
   }
};

void test_class_template(Q<R> q) {
   q.getArray();
   q.setArray(nullptr);
}

struct AnonSFields {
 struct {
  [[clang::unsafe_buffer_usage]]
  int a;
 };
};

void test_anon_struct_fields(AnonSFields anon) {
  int val = anon.a; //expected-warning{{field 'a' prone to unsafe buffer manipulation}}
}

union Union {
  [[clang::unsafe_buffer_usage]]
  int *ptr1;

  int ptr2;
};

struct C {
  Union ptr;
};

void test_attribute_union(C c) {
  int *p = c.ptr.ptr1; //expected-warning{{field 'ptr1' prone to unsafe buffer manipulation}}

  int address = c.ptr.ptr2;
}

struct AnonFields2 { 
  [[clang::unsafe_buffer_usage]] 
  struct { 
    int a; 
  }; 
};

void test_anon_struct(AnonFields2 af) {
  int val = af.a; // No warning here, as the attribute is not explicitly attached to field 'a'
  val++;
}
