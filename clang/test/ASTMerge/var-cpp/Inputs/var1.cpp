
template <typename T>
constexpr T my_pi = T(3.1415926535897932385L);  // variable template

template <> constexpr char my_pi<char> = '3';   // variable template specialization

template <typename T>
struct Wrapper {
  template <typename U> static constexpr U my_const = U(1);
   // Variable template partial specialization with member variable.
  template <typename U> static constexpr U *my_const<const U *> = (U *)(0);
};

constexpr char a[] = "hello";

template <> template <>
constexpr const char *Wrapper<float>::my_const<const char *> = a;
