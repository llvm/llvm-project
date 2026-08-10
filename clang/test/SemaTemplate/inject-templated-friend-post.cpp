// RUN: %clang_cc1 %s -std=c++98 -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -std=c++98 -triple %itanium_abi_triple -emit-llvm -o - -DPROTOTYPE | FileCheck --check-prefix=CHECK-PROTOTYPE %s
// RUN: %clang_cc1 %s -std=c++98 -triple %itanium_abi_triple -emit-llvm -o - -DINSTANTIATE | FileCheck --check-prefix=CHECK-INSTANTIATE %s
// RUN: %clang_cc1 %s -std=c++98 -triple %itanium_abi_triple -emit-llvm -o - -DPROTOTYPE -DINSTANTIATE | FileCheck --check-prefix=CHECK-PROTOTYPE-INSTANTIATE %s
// RUN: %clang_cc1 %s -DREDEFINE -verify
// RUN: %clang_cc1 %s -DPROTOTYPE -DREDEFINE -verify
// PR8007: friend function not instantiated, reordered version.
// Corresponds to http://gcc.gnu.org/bugzilla/show_bug.cgi?id=38392

// CHECK: define linkonce_odr{{.*}}_ZlsR11std_ostreamRK8StreamerI3FooE
// CHECK-PROTOTYPE: define linkonce_odr{{.*}}_ZlsR11std_ostreamRK8StreamerI3FooE
// CHECK-INSTANTIATE: define linkonce_odr{{.*}}_ZlsR11std_ostreamRK8StreamerI3FooE
// CHECK-PROTOTYPE-INSTANTIATE: define linkonce_odr{{.*}}_ZlsR11std_ostreamRK8StreamerI3FooE

struct std_ostream
{
  int dummy;
};

std_ostream cout;

template <typename STRUCT_TYPE>
struct Streamer;

typedef struct Foo {} Foo;

inline std_ostream& operator << (std_ostream&, const Streamer<Foo>&);

void test(const Streamer<Foo>& foo)
{
    cout << foo;
}

template <typename STRUCT_TYPE>
struct Streamer
{
    friend std_ostream& operator << (std_ostream& o, const Streamer& f) // expected-error{{redefinition of 'operator<<'}}
        {
            Streamer s(f);
            s(o);
            return o;
        }

    Streamer(const STRUCT_TYPE& s) : s(s) {}

    const STRUCT_TYPE& s;
    void operator () (std_ostream&) const;
};

#ifdef PROTOTYPE
std_ostream& operator << (std_ostream&, const Streamer<Foo>&);
#endif

#ifdef INSTANTIATE
template struct Streamer<Foo>;
#endif

#ifdef REDEFINE
std_ostream& operator << (std_ostream& o, const Streamer<Foo>&) // expected-note{{is here}}
{
  return o;
}
#endif

#ifndef INSTANTIATE
template <>
void Streamer<Foo>::operator () (std_ostream& o) const // expected-note{{requested here}}
{
}
#endif

int main(void)
{
    Foo foo;
    test(foo);
}

