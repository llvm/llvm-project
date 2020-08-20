// PR9073
template<typename _Tp>
class shared_ptr{
public:
  template<class _Alloc, class ..._Args>
  static
  shared_ptr<_Tp>
  allocate_shared(const _Alloc& __a, _Args&& ...__args);
};

template<class _Tp>
template<class _Alloc, class ..._Args>
shared_ptr<_Tp>
shared_ptr<_Tp>::allocate_shared(const _Alloc& __a, _Args&& ...__args)
{
  shared_ptr<_Tp> __r;
  return __r;
}

template<typename...Ts> struct outer {
  template<Ts...Vs, template<Ts> class ...Cs> struct inner {
    inner(Cs<Vs>...);
  };
};
template struct outer<int, int>;

template<typename ...T> void take_nondependent_pack(int (...arr)[sizeof(sizeof(T))]);

template<typename T> using hide = int;
template<typename ...T> void take_nondependent_pack_2(outer<hide<T>...>);
