
namespace std {
    struct destroying_delete_t { };
    template <class T> struct type_identity {
        using type = T;
    };
    typedef __SIZE_TYPE__ size_t;
    enum class align_val_t : size_t;
};

struct A {
    A();
   void *operator new(std::size_t);
   void operator delete(A*, std::destroying_delete_t);
};

struct B {
    B();
    void *operator new(std::type_identity<B>, std::size_t, std::align_val_t);
    void operator delete(std::type_identity<B>, void*, std::size_t, std::align_val_t);
};

struct C {
    C();
    template <class T> void *operator new(std::type_identity<T>, std::size_t, std::align_val_t);
    template <class T> void operator delete(std::type_identity<T>, void*, std::size_t, std::align_val_t);
};

struct D {
    D();
};
void *operator new(std::type_identity<D>, std::size_t, std::align_val_t);
void operator delete(std::type_identity<D>, void*, std::size_t, std::align_val_t);

struct E {
    E();
};
template <class T> void *operator new(std::type_identity<T>, std::size_t, std::align_val_t);
template <class T> void operator delete(std::type_identity<T>, void*, std::size_t, std::align_val_t);

void in_module_tests() {
  A* a = new A;
  delete a;
  B *b = new B;
  delete b;
  C *c = new C;
  delete c;
  D *d = new D;
  delete d;
  E *e = new E;
  delete e;
}
