// RUN: %check_clang_tidy %s bugprone-reserved-identifier %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-reserved-identifier.Invert: true, \
// RUN:     bugprone-reserved-identifier.AllowedIdentifiers: "std;reference_wrapper;ref;^c?ref;type;get" \
// RUN:   }}' -- \
// RUN:   -I%S/Inputs/reserved-identifier \
// RUN:   -isystem %S/Inputs/reserved-identifier/system

namespace std {

void __f() {}

void f();
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: declaration uses identifier 'f', which is not a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: void __f();
struct helper {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: declaration uses identifier 'helper', which is not a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: struct __helper {};
struct Helper {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: declaration uses identifier 'Helper', which is not a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: struct _Helper {};
struct _helper2 {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: declaration uses identifier '_helper2', which is not a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: struct __helper2 {};

template <class _Tp>
class reference_wrapper {
public:
  typedef _Tp type;

private:
  type *__f_;

public:
  reference_wrapper(type &__f)
      : __f_(&__f) {}
  // access
  operator type &() const { return *__f_; }
  type &get() const { return *__f_; }
};

template <class _Tp>
inline reference_wrapper<_Tp>
ref(_Tp &__t) noexcept {
  return reference_wrapper<_Tp>(__t);
}

template <class _Tp>
inline reference_wrapper<_Tp>
ref(reference_wrapper<_Tp> __t) noexcept {
  return ref(__t.get());
}

template <class Up>
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: declaration uses identifier 'Up', which is not a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: template <class _Up>
inline reference_wrapper<const Up>
cref(const Up &u) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: declaration uses identifier 'u', which is not a reserved identifier [bugprone-reserved-identifier]
  // CHECK-FIXES: cref(const _Up &__u) noexcept {
  return reference_wrapper<const Up>(u);
}

template <class _Tp>
inline reference_wrapper<_Tp>
cref(reference_wrapper<const _Tp> __t) noexcept {
  return cref(__t.get());
}

} // namespace std
