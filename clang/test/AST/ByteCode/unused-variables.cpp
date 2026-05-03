// RUN: %clang_cc1 -std=c++23 -Wunused -fexperimental-new-constant-interpreter -verify=both,expected %s
// RUN: %clang_cc1 -std=c++23 -Wunused                                         -verify=both,ref      %s


// both-no-diagnostics
namespace BaseUninitializedField {
  struct __optional_storage_base {
    int value;
    template <class _UArg> constexpr __optional_storage_base(_UArg) {}
  };

  struct optional : __optional_storage_base {
    template <class _Up>
    constexpr optional(_Up &&) : __optional_storage_base(0) {}
  };
  int main_x;
  void test() { optional opt{main_x}; }
}


namespace BaseInvalidLValue {
  int *addressof(int &);
  struct in_place_t {
  } in_place;
  template <class> struct __optional_storage_base {
    int *__value_;
    template <class _UArg>
    constexpr __optional_storage_base(in_place_t, _UArg &&__uarg) {
      int &__trans_tmp_1(__uarg);
      int &__val = __trans_tmp_1;
      int &__r(__val);
      __value_ = addressof(__r);
    }
  };
  template <class _Tp>
  struct __optional_copy_base : __optional_storage_base<_Tp> {
    using __optional_storage_base<_Tp>::__optional_storage_base;
  };
  template <class _Tp> struct __optional_move_base : __optional_copy_base<_Tp> {
    using __optional_copy_base<_Tp>::__optional_copy_base;
  };
  template <class _Tp>
  struct __optional_copy_assign_base : __optional_move_base<_Tp> {
    using __optional_move_base<_Tp>::__optional_move_base;
  };
  template <class _Tp>
  struct __optional_move_assign_base : __optional_copy_assign_base<_Tp> {
    using __optional_copy_assign_base<_Tp>::__optional_copy_assign_base;
  };
  struct optional : __optional_move_assign_base<int> {
    template <class _Up>
    constexpr optional(_Up &&__v) : __optional_move_assign_base(in_place, __v) {}
  };
  int test() {
    int x;
    /// With -Wunused, we will call EvaluateAsInitializer() on the variable here and if that
    /// succeeds, it will be reported unused. It should NOT succeed because the __value_ is an
    /// invalid lvalue.
    optional opt{x};
    return 0;
  }
}

namespace NonConstantInitChecksLValue {
  template <class _Tp, class>
  concept __weakly_equality_comparable_with = requires(_Tp __t) { __t; };
  template <class _Ip>
  concept input_or_output_iterator = requires(_Ip __i) { __i; };
  template <class _Sp, class _Ip>
  concept sentinel_for = __weakly_equality_comparable_with<_Sp, _Ip>;

  template <input_or_output_iterator _Iter, sentinel_for<_Iter> _Sent = _Iter>
  struct subrange {
    _Iter __begin_;
    _Sent __end_;
    constexpr subrange(auto, _Sent __sent) : __begin_(), __end_(__sent) {}
  };
  struct forward_iterator {
    int *it_;
  };
  void test() {
    using Range = subrange<forward_iterator>;
    int buffer[]{};
    /// The EvaluateAsInitializer() call needs to check the LValue and not just the lvalue fields.
    Range input(forward_iterator{}, {buffer});
  }
}

namespace PtrInBase {
  int *addressof(int &);
  template <class> struct __optional_storage_base {
    int *__value_;
    template <class _UArg>
    constexpr __optional_storage_base(_UArg &&__uarg) {
      int &__trans_tmp_1(__uarg);
      int &__val = __trans_tmp_1;
      int &__r(__val);
      __value_ = addressof(__r);
    }
  };
  struct optional : __optional_storage_base<int> {
    template <class _Up>
    constexpr optional(_Up &&__v) : __optional_storage_base(__v) {}
  };
  void test() {
    int x;
    optional opt{x};
  }
}
