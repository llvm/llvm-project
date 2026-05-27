// TDD tests for null checker features not yet implemented.
// Tests marked XFAIL-GAP document known gaps. Remove the workaround
// annotations as each gap is closed, and move the passing test to the
// appropriate main test file.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 -Wno-unused-value -Wno-nonnull %s -verify

#pragma clang assume_nonnull begin

// Forward declarations
int *_Nullable GetNullable();
int *_Nonnull GetNonnull();

// ==========================================================================
// GAP 1: Nonnull field nullable at method exit
// A method that nulls a _Nonnull field without restoring it.
// STATUS: NOT IMPLEMENTED
// ==========================================================================

struct OwnerWithNonnullField {
    int *_Nonnull field_;

    OwnerWithNonnullField(int *_Nonnull p) : field_(p) {}

    void steal_field() {
        field_ = nullptr; // expected-warning{{assigning nullable pointer to nonnull member}}
                          // expected-note@-1{{add a null check}}
        // XFAIL-GAP: should ALSO warn "nonnull field is nullable at method exit"
    }

    void swap_ok(int *_Nonnull replacement) {
        int *old = field_;
        field_ = replacement; // no warning — field is still nonnull at exit
    }
};

// ==========================================================================
// GAP 2: Nonnull default argument with null value
// STATUS: IMPLEMENTED
// ==========================================================================

void default_arg_null(int *_Nonnull p = nullptr);       // expected-warning{{nullable default argument for nonnull parameter 'p'}}
                                                         // expected-note@-1{{remove '_Nonnull' from the parameter, or change the default to a nonnull value}}
void default_arg_nullable(int *_Nonnull p = GetNullable()); // expected-warning{{nullable default argument for nonnull parameter 'p'}}
                                                             // expected-note@-1{{remove '_Nonnull' from the parameter, or change the default to a nonnull value}}
void default_arg_cast_nullable(int *_Nonnull p = (int*)GetNullable()); // expected-warning{{nullable default argument for nonnull parameter 'p'}}
                                                                        // expected-note@-1{{remove '_Nonnull' from the parameter, or change the default to a nonnull value}}
void default_arg_ok(int *_Nonnull p = GetNonnull()); // no warning

// ==========================================================================
// GAP 3: Nonnull member default initializer from nullable function
// STATUS: IMPLEMENTED
// ==========================================================================

struct NonnullMemberDefaultInit {
    int *_Nonnull p = nullptr;        // expected-warning{{initializing nonnull member 'p' with null}}
                                      // expected-note@-1{{remove '_Nonnull' if this member can be null, or remove the null initializer}}
    int *_Nonnull q = GetNullable();  // expected-warning{{initializing nonnull member 'q' with null}}
                                      // expected-note@-1{{remove '_Nonnull' if this member can be null, or remove the null initializer}}
    int *_Nonnull q2 = (int*)GetNullable(); // expected-warning{{initializing nonnull member 'q2' with null}}
                                             // expected-note@-1{{remove '_Nonnull' if this member can be null, or remove the null initializer}}
    int *_Nonnull r = GetNonnull();   // no warning
};

// ==========================================================================
// GAP 4: Const method return value caching
// if (obj.get()) { *obj.get(); } should be safe for const methods.
// STATUS: NOT IMPLEMENTED
// ==========================================================================

struct Holder {
    int *_Nullable ptr_;
    int *_Nullable get() const { return ptr_; }
    int *_Nullable get_nonconst() { return ptr_; }
};

void test_const_method_caching(const Holder& h) {
    if (h.get()) {
        // XFAIL-GAP: should NOT warn — same const method, same object
        (void)*h.get(); // expected-warning{{dereference of nullable pointer}}
                        // expected-note@-1{{add a null check}}
    }
}

void test_nonconst_method_warns(Holder& h) {
    if (h.get_nonconst()) {
        // Non-const: second call might return different value. Should warn.
        (void)*h.get_nonconst(); // expected-warning{{dereference of nullable pointer}}
                                 // expected-note@-1{{add a null check}}
    }
}

// ==========================================================================
// GAP 5: Smart pointer release() modeling
// release() returns the pointer and nulls out the smart ptr.
// STATUS: NOT IMPLEMENTED
// ==========================================================================

namespace std {
template <typename T>
struct unique_ptr {
    T* ptr;
    using pointer = T*;
    using element_type = T;
    pointer operator->() { return ptr; }
    element_type& operator*() { return *ptr; }
    pointer _Nullable get() { return ptr; }
    pointer _Nullable release() { pointer p = ptr; ptr = nullptr; return p; }
    explicit operator bool() const { return ptr != nullptr; }
    void reset() { ptr = nullptr; }
    void reset(T* p) { ptr = p; }
    explicit unique_ptr(T* p) : ptr(p) {}
    unique_ptr() : ptr(nullptr) {}
    unique_ptr(unique_ptr&& other) : ptr(other.ptr) { other.ptr = nullptr; }
    unique_ptr& operator=(unique_ptr&& other) { ptr = other.ptr; other.ptr = nullptr; return *this; }
    unique_ptr(const unique_ptr&) = delete;
    unique_ptr& operator=(const unique_ptr&) = delete;
};

template <typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args);

template <typename T>
T&& move(T& t) noexcept;
} // namespace std

struct Resource { int val; };

void test_release_nulls_smart_ptr() {
    auto sp = std::make_unique<Resource>();
    sp->val = 1; // OK — make_unique is nonnull

    Resource* raw = sp.release();
    raw->val = 2; // expected-warning{{dereference of nullable pointer}}
                   // expected-note@-1{{add a null check}}
                   // XFAIL-GAP: should NOT warn once release() is modeled

    sp->val = 3; // XFAIL-GAP: should warn — sp is null after release
}

// ==========================================================================
// GAP 6: Cross-object member nullable tracking (obj.member = nullptr)
// STATUS: WON'T FIX — requires NullableMembers<VarDecl*,FieldDecl*> set
// mirroring NarrowedMembers. Pattern is rare outside this-> context;
// this->member tracking already covers the common class-method case.
// ==========================================================================

struct CrossObjTarget { int val; };
struct CrossObj { CrossObjTarget *_Nonnull ptr; };

void cross_object_member_nullable(CrossObj obj) {
    obj.ptr = nullptr; // expected-warning{{assigning}}
                       // expected-note@-1{{add a null check}}
    obj.ptr->val = 1;  // XFAIL-GAP: should warn — obj.ptr is null
}

// ==========================================================================
// GAP 7: Aggregate init missing under-initialized _Nonnull fields
// STATUS: WON'T FIX — overlaps with -Wmissing-field-initializers.
// Zero-init of omitted fields is a language-level concern, not a
// flow-analysis concern.
// ==========================================================================

struct AggNonnull { int *_Nonnull a; int *_Nonnull b; };

void aggregate_underinit(int *_Nonnull p) {
    AggNonnull s = {p}; // XFAIL-GAP: should warn — b is zero-initialized to null
}

// ==========================================================================
// GAP 8: std::move from non-this member smart pointer
// STATUS: WON'T FIX — requires cross-object member narrowing (GAP 6).
// auto local = std::move(obj.sp) doesn't inherit narrowing onto local
// because NarrowedMembers isn't consulted in the move-construct handler.
// ==========================================================================

void move_from_non_this_member() {
    struct Holder { std::unique_ptr<Resource> sp; };
    Holder h;
    h.sp = std::make_unique<Resource>();
    auto local = std::move(h.sp);
    local->val = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
                     // XFAIL-GAP: should not warn — source was narrowed
}

#pragma clang assume_nonnull end
