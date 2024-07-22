class _Nullable Smart;

void f1(int * _Nonnull);

void f2(Smart); // OK, not required on smart-pointer types
using Alias = Smart;
void f3(Alias);

template <class T> class _Nullable SmartTmpl;
void f2(SmartTmpl<int>);
template <class T> void f2(SmartTmpl<T>);
