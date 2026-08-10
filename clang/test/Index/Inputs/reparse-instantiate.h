template <typename T> struct S;

template<typename T> void c(T)
{
}

template <> struct S <int>
{
  void a()
  {
    c(&S<int>::b);
  }
  void b() {}
};
