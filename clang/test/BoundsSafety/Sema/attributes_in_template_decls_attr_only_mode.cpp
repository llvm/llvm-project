// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -verify %s
#include <ptrcheck.h>

// Note: RefParamMustBePtrBad and RefParamMustBePtrGood should have identical
// type shape. They only have different names for the purposes of diagnostic
// reporting.

// Only use this class template with `T` being non-pointer types so that all
// errors appear here.
template <class T>
class RefParamMustBePtrBad {
    public:
    T __single ptr0; // expected-error 3{{single attribute only applies to pointer arguments}}
    T __unsafe_indexable ptr1; // expected-error 3{{unsafe_indexable attribute only applies to pointer arguments}}

    T __single get_single() const { // expected-error 3{{single attribute only applies to pointer arguments}}
        T tmp = ptr0;
        return tmp;
    }

    T __unsafe_indexable get_unsafe_indexable() const { // expected-error 3{{unsafe_indexable attribute only applies to pointer arguments}}
        T tmp = ptr1;
        return tmp;
    }

    void set_single(T __single v) { // expected-error 3{{single attribute only applies to pointer arguments}}
        ptr0 = v;
    }

    // Note: An error only shows up in the method bodies that use `T` directly
    // when this class is explicitly instantiated, e.g.:
    //
    // ```
    // template class RefParamMustBePtrGood<float*>;
    // ```
    // 
    // this instantiates the type and **all** its methods.
    //
    // For implicit instantiation, e.g. like this
    //
    // ```
    // void Instantiate_RefParamMustBePtrBad() {
    //   RefParamMustBePtrBad<void> bad0;
    //   bad0.set_single();
    // }
    // ```
    //
    // Method bodies are not instantiated, unless there is a call to the method.
    // However, the instantiation won't happen if the object (`bad0`) has a type
    // error. This type will have an error if `T` is not a pointer. So calls
    // to `set_single()` and `set_unsafe_indexable()` will not trigger template
    // instantiation and thus there won't be associated diagnostics.
    void set_single() {
        T __single tmp = nullptr; // expected-error{{single attribute only applies to pointer arguments}}
        ptr0 = tmp;
    }

    void set_unsafe_indexable() {
        T __unsafe_indexable tmp = nullptr; // expected-error{{unsafe_indexable attribute only applies to pointer argument}}
        ptr1 = tmp;
    }

    void set_unsafe_indexable(T __unsafe_indexable v) { // expected-error 3{{unsafe_indexable attribute only applies to pointer argument}}
        ptr0 = v;
    }

};

// Only use this class template with `T` being pointer types so it's clearer
// that the correct template instantiations work (i.e. there are no errors in
// this class).
template <class T>
class RefParamMustBePtrGood {
    public:
    T __single ptr0;
    T __unsafe_indexable ptr1;

    T __single get_single() const {
        T tmp = ptr0;
        return tmp;
    }

    T __unsafe_indexable get_unsafe_indexable() const {
        T tmp = ptr1;
        return tmp;
    }

    void set_single(T __single v) {
        ptr0 = v;
    }
    
    void set_single() {
        T __single tmp = nullptr;
        ptr0 = tmp;
    }

    void set_unsafe_indexable() {
        T __unsafe_indexable tmp = nullptr;
        ptr1 = tmp;
    }

    void set_unsafe_indexable(T __unsafe_indexable v) {
        ptr0 = v;
    }
};

// Do not instantiate this class
template <class T>
class RefParamMustBePtrNeverInstantiated {
    public:
    T __single ptr0;
    T __unsafe_indexable ptr1;

    T __single get_single() const {
        T tmp = ptr0;
        return tmp;
    }

    T __unsafe_indexable get_unsafe_indexable() const {
        T tmp = ptr1;
        return tmp;
    }

    void set_single(T __single v) {
        ptr0 = v;
    }
    
    void set_single() {
        T __single tmp = nullptr;
        ptr0 = tmp;
    }

    void set_unsafe_indexable() {
        T __unsafe_indexable tmp = nullptr;
        ptr1 = tmp;
    }

    void set_unsafe_indexable(T __unsafe_indexable v) {
        ptr0 = v;
    }
};

using PtrTypedef = char*;

// Explicit instantiation with good type
template
class RefParamMustBePtrGood<float*>;

void Instantiate_RefParamMustBePtrGood() {
    // Implicit instantiation
    RefParamMustBePtrGood<int*> good0;
    RefParamMustBePtrGood<PtrTypedef> good1;
}


// Explicit instantiation with bad type
// expected-note@+4{{in instantiation of template class 'RefParamMustBePtrBad<float>' requested here}}
// expected-note@+3{{in instantiation of member function 'RefParamMustBePtrBad<float>::set_single' requested here}}
// expected-note@+2{{in instantiation of member function 'RefParamMustBePtrBad<float>::set_unsafe_indexable' requested here}}
template
class RefParamMustBePtrBad<float>;

// Forward declaration does not trigger template instantiation
template <>
class RefParamMustBePtrBad<double>;

void Instantiate_RefParamMustBePtrBad() {
    // expected-note@+1{{in instantiation of template class 'RefParamMustBePtrBad<void>' requested here}}
    RefParamMustBePtrBad<void> bad0;

    // Note: In principle these calls should force the method bodies to be
    // instantiated. However, given that the `RefParamMustBePtrBad<void>` type
    // has errors that probably prevents instantiation.
    bad0.set_single();
    bad0.set_unsafe_indexable();

    // expected-note@+1{{in instantiation of template class 'RefParamMustBePtrBad<int>' requested here}}
    RefParamMustBePtrBad<int> bad1;
}

template <class T>
class RefParamIsPointee {
    public:
    T* __single ptr0;
    T* __unsafe_indexable pt1;
};

// Explicit instantiation
template class RefParamIsPointee<float>;

void Instantiate_RefParamIsPointee() {
    // Implicit instantiation
    RefParamIsPointee<int> good0;
    RefParamIsPointee<PtrTypedef> good1;
}

// =============================================================================
// T in method body
// =============================================================================
template <class T>
class TInMethodBodyGood {
    public:
    void test() {
        T __single tmp;
        tmp = nullptr;
    }
};

template <class T>
class TInMethodBodyBad {
    public:
    void test() {
        T __single tmp; // expected-error 2{{single attribute only applies to pointer arguments}}
        tmp = nullptr;
    }
};

// Explicit instantiation
template class TInMethodBodyGood<float*>;

void Instantiate_TInMethodBodyGood() {
    // Implicit instantiation
    TInMethodBodyGood<int*> good0;
    good0.test();
}

// Explicit instantiation
// expected-note@+1 {{in instantiation of member function 'TInMethodBodyBad<char>::test' requested here}}
template class TInMethodBodyBad<char>;

// Forward declaration does not trigger instantiation.
template <>
class TInMethodBodyBad<double>;

void Instantiate_TInMethodBodyBad() {
    // Implicit instantiation
    TInMethodBodyBad<int> bad0;
    // Note a call to `test()` is required to have the method body instantiated.
    // expected-note@+1{{in instantiation of member function 'TInMethodBodyBad<int>::test' requested here}}
    bad0.test();

    TInMethodBodyBad<float> bad1;
    // No call to `test()` so the method body is not instantiated.
}


// =============================================================================
// Partial specialization
// =============================================================================

template <class T, class U, class V>
class RefParamMustBePtrBadPartialBase {
    public:
    T __single ptr0; // expected-error 2{{single attribute only applies to pointer arguments}}
    U __unsafe_indexable ptr1; // expected-error 2{{unsafe_indexable attribute only applies to pointer arguments}}
    V counter;

    T __single get_ptr0() const { return ptr0; } // expected-error 2{{single attribute only applies to pointer arguments}}
    U __unsafe_indexable get_ptr1() const { return ptr1; } // expected-error 2{{unsafe_indexable attribute only applies to pointer arguments}}
    V get_counter() const { return counter; }

    void useT() const {
        // Note: An error would show up here if this method body was instantiated
        // with an invalid `T`. Explicit instantiation of a sub-class (e.g.
        // `RefParamMustBePtrBadPartialT`)  with an invalid `T` doesn't seem to
        // trigger instantiation of the parent class methods if the parent class
        // has type errors. Implicit instantiation (see
        // `Instantiate_RefParamMustBePtrBadPartial`) with calls to this method
        // doesn't seem to instantiate this method body due to errors on the
        // instantiated type.
        T __single tmp = ptr0;
    }

    void useU() const {
        // Note: An error would show up here if this method body was instantiated
        // with an invalid `T`. Explicit instantiation of a sub-class (e.g.
        // `RefParamMustBePtrBadPartialT`)  with an invalid `T` doesn't seem to
        // trigger instantiation of the parent class methods if the parent class
        // has type errors. Implicit instantiation (see
        // `Instantiate_RefParamMustBePtrBadPartial`) with calls to this method
        // doesn't seem to instantiate this method body due to errors on the
        // instantiated type.
        T __unsafe_indexable tmp = ptr1;
    }
};

template <class T, class U, class V>
class RefParamMustBePtrGoodPartialBase {
    public:
    T __single ptr0;
    U __unsafe_indexable ptr1;
    V counter;

    T __single get_ptr0() const { return ptr0; }
    U __unsafe_indexable get_ptr1() const { return ptr1; }
    V get_counter() const { return counter; }

    void useT() const {
        T __single tmp = ptr0;
    }

    void useU() const {
        T __unsafe_indexable tmp = ptr1;
    }
};

// good
template <class T>
class RefParamMustBePtrGoodPartialT : public RefParamMustBePtrGoodPartialBase<T, int*, int> {
    public:
    T __single ptr2;
    typeof(RefParamMustBePtrGoodPartialBase<T, int*, int>::ptr0) ptr3;
    T __single another_method() { return ptr2; }
};

template <class U>
class RefParamMustBePtrGoodPartialU : public RefParamMustBePtrGoodPartialBase<int*, U, int> {
    public:
    U __unsafe_indexable ptr2;
    typeof(RefParamMustBePtrGoodPartialBase<int*, U, int>::ptr1) ptr3;
    U __unsafe_indexable another_method() { return ptr2; }
};

// This partial specialization is never instantiated so it doesn't produce errors.
template <class V>
class RefParamMustBePtrGoodPartialV : public RefParamMustBePtrGoodPartialBase<int, int, V> {
    public:
    V __single ptr2;
    typeof(RefParamMustBePtrGoodPartialBase<int, int, V>::ptr0) ptr3;
    V __single another_method() { return ptr2; }
};

// Explicit instantiation
template
class RefParamMustBePtrGoodPartialT<float*>;
template
class RefParamMustBePtrGoodPartialU<float*>;

void Instantiate_RefParamMustBePtrGoodPartial() {
    // Implicit instantiation
    RefParamMustBePtrGoodPartialT<int*> good0;
    RefParamMustBePtrGoodPartialU<int*> good1;
}

// bad
// expected-note@+3{{in instantiation of template class 'RefParamMustBePtrBadPartialBase<float, int *, int>' requested here}}
// expected-note@+2{{in instantiation of template class 'RefParamMustBePtrBadPartialBase<int, int *, int>' requested here}}
template <class T>
class RefParamMustBePtrBadPartialT : public RefParamMustBePtrBadPartialBase<T, int*, int> {
    public:
    T __single ptr2; // expected-error 2{{single attribute only applies to pointer arguments}}
    // `ptr3` doesn't seem to get an error diagnostic. That's probably because
    // `ptr0` has errors.
    typeof(RefParamMustBePtrBadPartialBase<T, int*, int>::ptr0) ptr3;

    T __single another_method() { return ptr2; } // expected-error 2{{single attribute only applies to pointer arguments}}
};

// expected-note@+3 {{in instantiation of template class 'RefParamMustBePtrBadPartialBase<int *, float, int>' requested here}}
// expected-note@+2 {{in instantiation of template class 'RefParamMustBePtrBadPartialBase<int *, int, int>' requested here}}
template <class U>
class RefParamMustBePtrBadPartialU : public RefParamMustBePtrBadPartialBase<int*, U, int> {
    public:
    U __unsafe_indexable ptr2; // expected-error 2{{unsafe_indexable attribute only applies to pointer arguments}}
    // `ptr3` doesn't seem to get an error diagnostic. That's probably because
    // `ptr0` has errors.
    typeof(RefParamMustBePtrBadPartialBase<int*, U, int>::ptr1) ptr3;
    U __unsafe_indexable another_method() { return ptr2; } // expected-error 2{{unsafe_indexable attribute only applies to pointer arguments}}
};

// A partial specialization that is never instantiated will produce errors for
// concrete types in the specialization.
template <class V>
class RefParamMustBePtrBadPartialV : public RefParamMustBePtrBadPartialBase<int, int, V> {
    public:
    int __single ptr2; // expected-error{{'__single' attribute only applies to pointer arguments}}
    typeof(RefParamMustBePtrGoodPartialBase<int, int, V>::ptr0) ptr3;
    int __single another_method() { return ptr2; } // expected-error{{'__single' attribute only applies to pointer arguments}}
    V __single other; // no error
};

// explicit instantiation
// expected-note@+1 2{{in instantiation of template class 'RefParamMustBePtrBadPartialT<float>' requested here}}
template class RefParamMustBePtrBadPartialT<float>;
// expected-note@+1 2{{in instantiation of template class 'RefParamMustBePtrBadPartialU<float>' requested here}}
template class RefParamMustBePtrBadPartialU<float>;

void Instantiate_RefParamMustBePtrBadPartial() {
    // Implicit instantiation
    // expected-note@+1 2{{in instantiation of template class 'RefParamMustBePtrBadPartialT<int>' requested here}}
    RefParamMustBePtrBadPartialT<int> bad0;
    // These calls don't seem to be trigger instantiation of the method body
    // due to `bad0` having type errors.
    bad0.useT();
    bad0.useU();
    // expected-note@+1 2{{in instantiation of template class 'RefParamMustBePtrBadPartialU<int>' requested here}}
    RefParamMustBePtrBadPartialU<int> bad1;
    // These calls don't seem to be trigger instantiation of the method body
    // due to `bad1` having type errors.
    bad1.useT();
    bad1.useU();
}

// external counted attributes

// Note: RefParamMustBePtrExternallyCountedGood and
// RefParamMustBePtrExternallyCountedBad should have identical type shape. They
// only have different names for the purposes of diagnostic reporting.
template <class T>
class RefParamMustBePtrExternallyCountedGood {
    public:
    int size;
    T end_ptr;
    T __counted_by(size) cb;
    T __counted_by_or_null(size) cbon;
    T __sized_by(size) sb;
    T __sized_by_or_null(size) sbon;
    T __ended_by(end_ptr) eb;

    T __counted_by(size) ret_cb() {
        return cb;
    }

    // FIXME: This produces an error diagnostic before instantiating templates
    // but it shouldn't.
    // rdar://152538978.
    // void cb_param(T __counted_by(size) ptr, int size) {}

    void useT() {
        int size_local = size;
        T __counted_by(size_local) tmp = cb;
    }
};

// Explicit instantiation
template class RefParamMustBePtrExternallyCountedGood<float*>;

void Instantiate_RefParamMustBePtrExternallyCountedGood() {
    // Implicit instantiation
    RefParamMustBePtrExternallyCountedGood<int*> good0;
    RefParamMustBePtrExternallyCountedGood<PtrTypedef> good1;
}

template <class T>
class RefParamMustBePtrExternallyCountedBad {
    public:
    int size;
    T end_ptr;
    T __counted_by(size) cb; // expected-error 2{{'counted_by' attribute only applies to pointer arguments}}
    T __counted_by_or_null(size) cbon; // expected-error 2{{'counted_by_or_null' attribute only applies to pointer arguments}}
    T __sized_by(size) sb; // expected-error 2{{'sized_by' attribute only applies to pointer arguments}}
    T __sized_by_or_null(size) sbon; // expected-error 2{{'sized_by_or_null' attribute only applies to pointer arguments}}
    T __ended_by(end_ptr) eb; // expected-error 2{{'ended_by' attribute requires a pointer type argument}}

    T __counted_by(size) ret_cb() { // expected-error 2{{'counted_by' attribute only applies to pointer arguments}}
        return cb;
    }

    // expected-error@+1{{'__counted_by' attribute only applies to pointer arguments}}
    void cb_param(T __counted_by(size) ptr, int size) {}

    void useT() {
        int size_local = size;
        T __counted_by(size_local) tmp = cb; // expected-error{{'counted_by' attribute only applies to pointer arguments}}
    }
};

// Explicit instantiation
// expected-note@+2{{in instantiation of member function 'RefParamMustBePtrExternallyCountedBad<float>::useT' requested here}}
// expected-note@+1{{in instantiation of template class 'RefParamMustBePtrExternallyCountedBad<float>' requested here}}
template class RefParamMustBePtrExternallyCountedBad<float>;

void Instantiate_RefParamMustBePtrExternallyCountedBad() {
    // Implicit instantiation
    // expected-note@+1 {{in instantiation of template class 'RefParamMustBePtrExternallyCountedBad<int>' requested here}}
    RefParamMustBePtrExternallyCountedBad<int> bad0;
}

// Attributes that currently don't work

#define __bidi_indexable __attribute__((__bidi_indexable__))
#define __indexable __attribute__((__indexable__))

template <class T>
class RefParamUnsupportedAttrs {
    public:
    // TODO: Support __null_terminated (rdar://152451848).
    T __null_terminated ptr0; // expected-error{{'__terminated_by' attribute can be applied to pointers, constant-length arrays or incomplete arrays}}
    T __bidi_indexable ptr1;  // expected-warning{{'__bidi_indexable__' attribute ignored}}
    T __indexable ptr2; // expected-warning{{'__indexable__' attribute ignored}}
};

// Explicit instantiation
// FIXME: Explicit instantiation with an invalid T should generate diagnostics.
template class RefParamUnsupportedAttrs<float>;

void Instantiate_RefParamUnsupportedAttrs() {
    // Implicit instantiation
    RefParamUnsupportedAttrs<int*> broken;
}

template <class T>
class RefParamPtrUnsupportedAttrs {
    public:
    // TODO: Support __null_terminated (rdar://152451848).
    // FIXME: This diagnostic looks wrong.
    T* __null_terminated ptr0; // expected-error{{pointee type of pointer with '__terminated_by' attribute must be an integer or a non-wide pointer}}
    T* __bidi_indexable ptr1;  // expected-warning{{'__bidi_indexable__' attribute ignored}}
    T* __indexable ptr2; // expected-warning{{'__indexable__' attribute ignored}}
};

// Explicit instantiation
// FIXME: Explicit instantiation with an invalid T should generate diagnostics.
template class RefParamPtrUnsupportedAttrs<float>;

void Instantiate_RefParamPtrUnsupportedAttrs() {
    // Implicit instantiation
    RefParamPtrUnsupportedAttrs<int> ok;
}
