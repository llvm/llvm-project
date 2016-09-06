namespace B {

  template <class _CharT>
  struct basic_ostream {
    basic_ostream& operator<<(basic_ostream& (*__pf)());
  };


  template <class _CharT> basic_ostream<_CharT>&
  endl();

  struct S1 {
    template <class _CharT> friend void
    operator<<(basic_ostream<_CharT>& __os, const S1& __x);
  };

  S1 setw(int __n);

  template <class _CharT> class S2;

  template <class _CharT> void
  operator<<(basic_ostream<_CharT>& __os, const S2<_CharT>& __x);

  template <class _CharT>
  struct S2 {
    template <class _Cp> friend void
    operator<<(basic_ostream<_Cp>& __os, const S2<_Cp>& __x);
  };

}
