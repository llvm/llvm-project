#include "__config"

_LIBCPP_BEGIN_NAMESPACE_STD

// Test that _LIBCPP_VERSION and __libcpp_version are in sync.
// The __libcpp_version file stores only a number representing the libc++
// version so it can be easily parsed by clang.
static_assert(_LIBCPP_VERSION ==
#include "__libcpp_version"
    , "version file does not match");

int __libcpp_library_version() { return _LIBCPP_VERSION; }

_LIBCPP_END_NAMESPACE_STD
