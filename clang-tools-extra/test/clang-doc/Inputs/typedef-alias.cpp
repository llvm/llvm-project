/// \brief This is u_long
using u_long = unsigned long;

/// \brief This is IntPtr
typedef int *IntPtr;

template <typename T> class Vector {
  /// \brief This is a Ptr
  using Ptr = IntPtr;
};

template <typename T> using Vec = Vector<T>;

using IntVec = Vector<int>;
