namespace A {
inline
namespace __1 {
  template <class _Tp> class allocator;
  template <class _Tp, class _Alloc = allocator<_Tp>> class list;
  template <class _VoidPtr> class __list_iterator {
    template <class> friend class list;
    template <class, class> friend class list;
  };
  template <class _Tp, class _Alloc> class __list_imp {};
  template <class _Tp, class _Alloc> class list : __list_imp<_Tp, _Alloc> {
  public:
    list() {}
  };
  template <class _Tp> void f(list<_Tp>);
}
}
