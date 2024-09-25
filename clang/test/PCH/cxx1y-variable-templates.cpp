// No PCH:
// RUN: %clang_cc1 -pedantic -std=c++1y -include %s -include %s -verify %s -DNONPCH
// RUN: %clang_cc1 -pedantic -std=c++1y -include %s -include %s -verify %s -DNONPCH -DERROR
//
// With PCH:
// RUN: %clang_cc1 -pedantic -std=c++1y -emit-pch %s -o %t.a -DHEADER1
// RUN: %clang_cc1 -pedantic -std=c++1y -include-pch %t.a -emit-pch %s -o %t.b -DHEADER2
// RUN: %clang_cc1 -pedantic -std=c++1y -include-pch %t.b -verify %s -DHEADERUSE

// RUN: %clang_cc1 -pedantic -std=c++1y -emit-pch -fpch-instantiate-templates %s -o %t.a -DHEADER1
// RUN: %clang_cc1 -pedantic -std=c++1y -include-pch %t.a -emit-pch -fpch-instantiate-templates %s -o %t.b -DHEADER2
// RUN: %clang_cc1 -pedantic -std=c++1y -include-pch %t.b -verify %s -DHEADERUSE

#ifndef ERROR
// expected-no-diagnostics
#endif

#ifdef NONPCH
#if !defined(HEADER1)
#define HEADER1
#undef HEADER2
#undef HEADERUSE
#elif !defined(HEADER2)
#define HEADER2
#undef HEADERUSE
#else
#define HEADERUSE
#undef HEADER1
#undef HEADER2
#endif
#endif


// *** HEADER1: First header file
#if defined(HEADER1) && !defined(HEADER2) && !defined(HEADERUSE)

template<typename T> T var0a = T();
template<typename T> extern T var0b;

namespace join {
  template<typename T> T va = T(100);
  template<typename T> extern T vb;

  namespace diff_types {
#ifdef ERROR
    template<typename T> extern float err0;
    template<typename T> extern T err1;
#endif
    template<typename T> extern T def;
  }

}

namespace spec {
  template<typename T> constexpr T va = T(10);
  template<> constexpr float va<float> = 1.5;
  template constexpr int va<int>;

  template<typename T> T vb = T();
  template<> constexpr float vb<float> = 1.5;

  template<typename T> T vc = T();

  template<typename T> constexpr T vd = T(10);
  template<typename T> T* vd<T*> = new T();
}

namespace spec_join1 {
  template<typename T> T va = T(10);
#ifdef ERROR
  template<> float va<float>; // expected-note {{previous definition is here}}
#endif
  extern template int va<int>;

  template<typename T> T vb = T(10);
#ifdef ERROR
  template<> float vb<float>; // expected-note {{previous definition is here}}
#endif

  template<typename T> T vc = T(10);

  template<typename T> T vd = T(10);
  template<typename T> extern T* vd<T*>;
}

#endif


// *** HEADER2: Second header file -- including HEADER1
#if defined(HEADER2) && !defined(HEADERUSE)

namespace join {
  template<typename T> extern T va;
  template<> constexpr float va<float> = 2.5;

  template<typename T> T vb = T(100);

  namespace diff_types {
#ifdef ERROR
    template<typename T> extern T err0; // expected-error {{redeclaration of 'err0' with a different type: 'T' vs 'float'}}  // expected-note@46 {{previous declaration is here}}
    template<typename T> extern float err1; // expected-error {{redeclaration of 'err1' with a different type: 'float' vs 'T'}} // expected-note@47 {{previous declaration is here}}
#endif
    template<typename T> extern T def;
  }
}

namespace spec_join1 {
  template<typename T> extern T va;
#ifdef ERROR
  template<> float va<float> = 1.5; // expected-error {{redefinition of 'va<float>'}}
#endif
  extern template int va<int>;

#ifdef ERROR
  template<> float vb<float> = 1.5; // expected-error {{redefinition of 'vb<float>'}}
#endif
  template int vb<int>;

  template<> float vc<float> = 1.5;
  template int vc<int>;

  template<typename T> extern T vd;
  template<typename T> T* vd<T*> = new T();
}

#endif

// *** HEADERUSE: File using both header files -- including HEADER2
#ifdef HEADERUSE

template int var0a<int>;
float fvara = var0a<float>;

template<typename T> extern T var0a;

template<typename T> T var0b = T();
template int var0b<int>;
float fvarb = var0b<float>;

namespace join {
  template const int va<const int>;
  template<> const int va<int> = 50;
  static_assert(va<float> == 2.5, "");
  static_assert(va<int> == 50, "");

  template<> constexpr float vb<float> = 2.5;
  template const int vb<const int>;
  static_assert(vb<float> == 2.5, "");
  static_assert(vb<const int> == 100, "");

  namespace diff_types {
    template<typename T> T def = T();
  }

}

namespace spec {
  static_assert(va<float> == 1.5, "");
  static_assert(va<int> == 10, "");

  template<typename T> T* vb<T*> = new T();
  int* intpb = vb<int*>;
  static_assert(vb<float> == 1.5, "");

  template<typename T> T* vc<T*> = new T();
  template<> constexpr float vc<float> = 1.5;
  int* intpc = vc<int*>;
  static_assert(vc<float> == 1.5, "");

  char* intpd = vd<char*>;
}

namespace spec_join1 {
  template int va<int>;
  int a = va<int>;

  template<typename T> extern T vb;
  int b = vb<int>;

  int* intpb = vd<int*>;
}

#endif
