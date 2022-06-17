namespace std {

void __f() {}

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

} // namespace std
