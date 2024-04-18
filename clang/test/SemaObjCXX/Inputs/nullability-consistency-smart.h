class _Nullable Smart;

void f1(int * _Nonnull);

void f2(Smart); // OK, not required on smart-pointer types
using Alias = Smart;
void f3(Alias);
