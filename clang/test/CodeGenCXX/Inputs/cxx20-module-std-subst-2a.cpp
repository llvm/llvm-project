module;
# 5 __FILE__ 1
namespace std {
template <typename A> struct allocator {};
template <typename C, typename T, typename A>
class basic_string;
} // namespace std
# 12 "" 2
export module RenameString;
export template <typename C, typename T>
using str = std::basic_string<C, T, std::allocator<C>>;
