// RUN: %clang_cc1 -fsyntax-only -Wnrvo -Wno-return-mismatch -verify %s
struct MyClass {
    int value;
    int c;
    MyClass(int v) : value(v), c(0) {}
    MyClass(const MyClass& other) : value(other.value) { c++; }
};

MyClass create_object(bool condition) {
    MyClass obj1(1);
    MyClass obj2(2);
    if (condition) {
        return obj1; // expected-warning{{not eliding copy on return}}
    }
    return obj2; // expected-warning{{not eliding copy on return}}
}

MyClass create_object2(){
    MyClass obj(1);
    return obj; // no warning
}

template<typename T>
T create_object3(){
    T obj(1);
    return obj; // no warning
}

// Known issue: if a function template uses a 
// deduced return type (i.e. auto or decltype(auto)), 
// then NRVO is not applied for any instantiation of 
// that function template 
// (see https://github.com/llvm/llvm-project/issues/95280).
template<typename T>
auto create_object4(){
    T obj(1);
    return obj; // expected-warning{{not eliding copy on return}}
}

template<bool F>
MyClass create_object5(){
    MyClass obj1(1);
    if constexpr (F){
        MyClass obj2(2);
        return obj2; // no warning
    }
  // Missed NRVO optimization by clang
  return obj1; // expected-warning{{not eliding copy on return}}
}

constexpr bool Flag = false;

MyClass create_object6(){
  MyClass obj1(1);
  if constexpr (Flag){
    MyClass obj2(2);
    return obj2; // expected-warning{{not eliding copy on return}}
  }
  return obj1; // no warning
}

void create_object7(){
    if constexpr (Flag){
        MyClass obj1(1);
        return obj1; // no warning
    }
}

void init_templates(){
    create_object3<MyClass>(); // no warning
    create_object4<MyClass>(); // expected-note {{in instantiation of function template specialization 'create_object4<MyClass>' requested here}}
    create_object5<false>(); // expected-note {{in instantiation of function template specialization 'create_object5<false>' requested here}}
}
