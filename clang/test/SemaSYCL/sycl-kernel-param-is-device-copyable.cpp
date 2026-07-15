// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -fsycl-is-host -verify %s
// RUN: %clang_cc1 -triple spirv64 -std=c++17 -fsyntax-only -fsycl-is-device -verify %s

namespace std {

template <bool B>
struct bool_constant {
  static constexpr bool value = B;
};

using true_type = bool_constant<true>;

template <typename T>
struct is_trivially_copyable : bool_constant<__is_trivially_copyable(T)> {};

template <typename T>
inline constexpr bool is_trivially_copyable_v = is_trivially_copyable<T>::value;

} // namespace std

// A unique kernel name type is required for each declared kernel entry point.
template<int, int = 0> struct KN;

// A generic kernel launch function.
template<typename KNT, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

namespace sycl {

template <typename T>
struct is_device_copyable : std::is_trivially_copyable<T> {};

template <typename T>
inline constexpr bool is_device_copyable_v = is_device_copyable<T>::value;

} // namespace sycl


class NotTriviallyCopyable {
public:
  NotTriviallyCopyable() {};
  NotTriviallyCopyable(const NotTriviallyCopyable& x);
};
static_assert(!std::is_trivially_copyable_v<NotTriviallyCopyable>,
  "NotTriviallyCopyable should be not std::is_trivially_copyable");

class DeviceCopyable : public NotTriviallyCopyable {};
template<>
struct sycl::is_device_copyable<DeviceCopyable> : std::true_type {};


// Check that sycl::is_device_copyable is respected
namespace iscopyable1 {
// Kernel entry point template definition.
template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void kernel_single_task(T t) {}

void test() {
  DeviceCopyable a;
  NotTriviallyCopyable b;
  kernel_single_task<KN<1>>(a);
  kernel_single_task<KN<2>>(b);
}
} // namespace iscopyable1