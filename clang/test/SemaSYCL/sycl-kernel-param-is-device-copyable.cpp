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

struct DefinitelyCopyable {
  int foo = 0;
};
static_assert(std::is_trivially_copyable_v<DefinitelyCopyable>,
  "DefinitelyCopyable should be trivially copyable");

// Check that sycl::is_device_copyable is respected
namespace iscopyable1 {
// Kernel entry point template definition.
template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void kernel_single_task(T t) {} // expected-error {{'NotTriviallyCopyable' is not device copyable (sycl::is_device_copyable) and cannot be used as a kernel parameter}}
                                // expected-note-re@-1 2{{within parameter 't' of type '{{.*}}' declared here}}

void test() {
  DefinitelyCopyable a;
  kernel_single_task<KN<1>>(a);

  NotTriviallyCopyable b;
  kernel_single_task<KN<2>>(b);
  // expected-note-re@-1 {{in instantiation of function template specialization 'iscopyable1::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}

  DeviceCopyable c;
  kernel_single_task<KN<3>>(c);


  kernel_single_task<KN<4>>([=] { (void) a; });

  kernel_single_task<KN<5>>([=] { (void) c; });

  kernel_single_task<KN<6>>([=] { (void) b; });
  // expected-error@-1 {{'NotTriviallyCopyable' is not device copyable (sycl::is_device_copyable) and cannot be used as a kernel parameter}}
  // expected-note-re@-2 {{in instantiation of function template specialization 'iscopyable1::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  // expected-note@-3 {{within capture 'b' of lambda expression here}}

  auto notCopyableLambda = [](NotTriviallyCopyable notCopyable) { (void) notCopyable; };
  kernel_single_task<KN<7>>(notCopyableLambda);
  kernel_single_task<KN<8>>([=](NotTriviallyCopyable NC) { notCopyableLambda(NC); });
  // TODO this isn't firing; shouldn't this create an error
  
}

} // namespace iscopyable1

struct HasNonDeviceCopyableMember {
  NotTriviallyCopyable member;
};

struct ExplicitlyDeviceCopyableWithMember {
  NotTriviallyCopyable member;
};
template<>
struct sycl::is_device_copyable<ExplicitlyDeviceCopyableWithMember>
    : std::true_type {};

namespace iscopyable2 {
template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void kernel_single_task(T t) {} // expected-note-re {{within parameter 't' of type '{{.*}}' declared here}}

void test() {
  HasNonDeviceCopyableMember a;
  kernel_single_task<KN<9>>([=] { (void) a; });
  // expected-error@-1 {{'HasNonDeviceCopyableMember' is not device copyable (sycl::is_device_copyable) and cannot be used as a kernel parameter}}
  // expected-note-re@-2 {{in instantiation of function template specialization 'iscopyable2::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  // expected-note@-3 {{within capture 'a' of lambda expression here}}

  ExplicitlyDeviceCopyableWithMember b;
  kernel_single_task<KN<10>>([=] { (void) b; });
}
} // namespace iscopyable2

