// clang-format off
// REQUIRES: lld

// Test that we can display function signatures with class types.
// RUN: clang-cl /Z7 /GS- /GR- /c -Xclang -fkeep-static-consts /Fo%t.obj -- %s
// RUN: lld-link /DEBUG /nodefaultlib /entry:main /OUT:%t.exe /PDB:%t.pdb -- %t.obj
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 lldb -f %t.exe -s \
// RUN:     %p/Inputs/function-types-classes.lldbinit | FileCheck %s

// This is just some unimportant helpers needed so that we can get reference and
// rvalue-reference types into return values.
template<typename T>
struct MakeResult {
  static T result() {
    return T{};
  }
};

template<typename T>
struct MakeResult<T&> {
  static T& result() {
    static T t;
    return t;
  }
};

template<typename T>
struct MakeResult<T&&> {
  static T&& result() {
    static T t;
    return static_cast<T&&>(t);
  }
};


template<typename R>
R nullary() { return MakeResult<R>::result(); }

template<typename R, typename A, typename B>
R three(A a, B b) { return MakeResult<R>::result(); }

template<typename R, typename A, typename B, typename C>
R four(A a, B b, C c) { return MakeResult<R>::result(); }

struct S {};
class C {};
union U {};
enum E {};

namespace A {
  namespace B {
    // NS::NS
    struct S { };
  }

  struct C {
    // NS::Struct
    struct S {};
  };
}

struct B {
  struct A {
    // Struct::Struct
    struct S {};
  };
};

template<typename T>
struct TC {};

// const and volatile modifiers
auto a = &four<S, C*, U&, E&&>;
// CHECK: (S (*)(C *, U &, E &&)) a = {{.*}}
auto b = &four<E, const S*, const C&, const U&&>;
// CHECK: (E (*)(const S *, const C &, const U &&)) b = {{.*}}
auto c = &four<U, volatile E*, volatile S&, volatile C&&>;
// CHECK: (U (*)(volatile E *, volatile S &, volatile C &&)) c = {{.*}}
auto d = &four<C, const volatile U*, const volatile E&, const volatile S&&>;
// CHECK: (C (*)(const volatile U *, const volatile E &, const volatile S &&)) d = {{.*}}

// classes nested in namespaces and inner classes

// FIXME: LLDB with native pdb plugin doesn't currently resolve nested names
// correctly, because it requires creating clang::NamespaceDecl or
// clang::RecordDecl for the outer namespace or classes.  PDB doesn't contain
// sufficient information to distinguish namespace scopes from nested class
// scopes, so the best we can hope for is a heuristic reconstruction of the
// clang AST based on demangling the type's unique name.  However, this is
// as-yet unimplemented in the native PDB plugin, so for now all of these will
// all just look like `S` when LLDB prints them.
auto e = &three<A::B::S*, B::A::S*, A::C::S&>;
// CHECK: (S *(*)(S *, S &)) e = {{.*}}
auto f = &three<A::C::S&, A::B::S*, B::A::S*>;
// CHECK: (S &(*)(S *, S *)) f = {{.*}}
auto g = &three<B::A::S*, A::C::S&, A::B::S*>;
// CHECK: (S *(*)(S &, S *)) g = {{.*}}

// parameter types that are themselves template instantiations.
auto h = &four<TC<void>, TC<int>, TC<TC<int>>, TC<A::B::S>>;
// Note the awkward space in TC<TC<int> >.  This is because this is how template
// instantiations are emitted by the compiler, as the fully instantiated name.
// Only via reconstruction of the AST through the mangled type name (see above
// comment) can we hope to do better than this).
// CHECK: (TC<void> (*)(TC<int>, TC<TC<int> >, S>)) h = {{.*}}

auto i = &nullary<A::B::S>;
// CHECK: (S (*)()) i = {{.*}}


// Make sure we can handle types that don't have complete debug info.
struct Incomplete;
auto incomplete = &three<Incomplete*, Incomplete**, const Incomplete*>;
// CHECK: (Incomplete *(*)(Incomplete **, const Incomplete *)) incomplete = {{.*}}

int main(int argc, char **argv) {
  return 0;
}
