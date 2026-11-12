BEGIN
template<typename T = void> struct L;
struct FriendL {
  template<typename T> friend struct L;
};
END

namespace DeferredLookup {
  namespace Indirect {
    template<typename, bool = true> struct A {};
    template<typename> struct B { template<typename T> using C = A<T>; };
  }
}
