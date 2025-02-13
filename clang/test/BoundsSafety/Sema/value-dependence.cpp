// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -verify %s
// FIXME: Compilation fails when return_size bounds check is enabled (rdar://150044760) so disable it for now.
// RUN: %clang_cc1 -fbounds-safety -fno-bounds-safety-bringup-missing-checks=return_size -fexperimental-bounds-safety-cxx -verify=expected,bounds-safety %s

#include <ptrcheck.h>

void * __sized_by_or_null(size) malloc(int size);

int n;
template <typename T>
T bar(int x) {
    // expected-error@+2{{argument of '__sized_by_or_null' attribute cannot refer to declaration from a different scope}}
    // expected-error@+1{{argument of '__sized_by_or_null' attribute cannot refer to declaration of a different lifetime}}
    int * __sized_by_or_null(sizeof(T) * n) p;
    p = static_cast<int*>(malloc(x));
    n = x;
    return *p;
}

template <typename T>
struct Outer {
    struct Inner {
        T size;
        T * __sized_by_or_null(size) p_m;
        // FIXME: `cannot extract the lower bound of 'int *' because it has no bounds specification`
        // error diagnostic is emitted here when new bounds checks are on (rdar://150044760).
        T * __sized_by_or_null(sizeof(T) * n) mymalloc_m(int n) {
            return static_cast<T *>(malloc(sizeof(T) * n));
        }

        void bar_m(int q) {
            // bounds-safety-error@+2{{assignment to 'int *__single __sized_by_or_null(size)' (aka 'int *__single') '->p_m' requires corresponding assignment to '->size'; add self assignment '->size = ->size' if the value has not changed}}
            // bounds-safety-note@#instantiation {{in instantiation of member function 'Outer<int>::Inner::bar_m' requested here}}
            this->p_m = mymalloc_m(q);
        }
    } inner;

    void foo_m(int m) {
        int l;
        T * __sized_by_or_null(l) p1;
        // bounds-safety-error@+2{{assignment to 'int *__single __sized_by_or_null(l)' (aka 'int *__single') 'p1' requires corresponding assignment to 'l'; add self assignment 'l = l' if the value has not changed}}
        // bounds-safety-note@#instantiation {{in instantiation of member function 'Outer<int>::foo_m' requested here}}
        p1 = inner.mymalloc_m(m);
    }
};

template class Outer<int>; // #instantiation

template <typename T>
struct OuterNoInstantiation {
    struct InnerNoInstantiation {
        T size;
        T * __sized_by_or_null(size) p_m;
        T * __sized_by_or_null(sizeof(T) * n) mymalloc_m(int n) {
            return static_cast<T *>(malloc(sizeof(T) * n));
        }

        void bar_m(int q) {
            this->p_m = mymalloc_m(q);
        }
    } inner;

    void foo_m(int m) {
        int l;
        T * __sized_by_or_null(l) p1;
        p1 = inner.mymalloc_m(m);
    }
};

