// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -fsycl-is-host -verify %s
// RUN: %clang_cc1 -triple spirv64 -std=c++17 -fsyntax-only -fsycl-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -fsycl-is-host -Wpedantic-sycl -DCHECK_PEDANTIC_SYCL -verify %s
// RUN: %clang_cc1 -triple spirv64 -std=c++17 -fsyntax-only -fsycl-is-device -Wpedantic-sycl -DCHECK_PEDANTIC_SYCL -verify %s

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

template<typename KNT, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

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

template<typename KNT, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

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

#ifdef CHECK_PEDANTIC_SYCL

struct BadDestructorDeleted {
  // Classes with deleted destructors are still is_trivially_copyable: This
  // copy constructor is added to make the class not trivially copyable.
  // FIXME: SYCL spec simultaneously stipulates deleted destructors are UB
  // for device-copyable classes while also stipulates that all
  // is_trivially_copyable classes are device-copyable. Which is it?
  BadDestructorDeleted(const BadDestructorDeleted &) {}
  ~BadDestructorDeleted() = delete;
  // expected-warning@-1 {{'BadDestructorDeleted' is explicitly marked as device copyable (sycl::is_device_copyable) but does not have a public, non-deleted destructor}}
};
template<>
struct sycl::is_device_copyable<BadDestructorDeleted> : std::true_type {};

struct BadDestructorPrivate {
  // Classes with deleted destructors are still is_trivially_copyable: This
  // copy constructor is added to make the class not trivially copyable.
  BadDestructorPrivate(const BadDestructorPrivate &) {}
private:
  ~BadDestructorPrivate() {}
  // expected-warning@-1 {{'BadDestructorPrivate' is explicitly marked as device copyable (sycl::is_device_copyable) but does not have a public, non-deleted destructor}}
};
template<>
struct sycl::is_device_copyable<BadDestructorPrivate> : std::true_type {};

struct PrivateCopyCtor {
  PrivateCopyCtor() {}
private:
  PrivateCopyCtor(const PrivateCopyCtor &) {}
  // expected-warning@-1 {{'PrivateCopyCtor' is explicitly marked as device copyable (sycl::is_device_copyable) but its eligible copy constructor is not public}}
};
template<>
struct sycl::is_device_copyable<PrivateCopyCtor> : std::true_type {};

struct PrivateMoveAssign {
  PrivateMoveAssign() {}
private:
  PrivateMoveAssign& operator=(PrivateMoveAssign &&other) { return other; }
  // expected-warning@-1 {{'PrivateMoveAssign' is explicitly marked as device copyable (sycl::is_device_copyable) but its eligible move assignment operator is not public}}
};
template<>
struct sycl::is_device_copyable<PrivateMoveAssign> : std::true_type {};

struct NoEligibleSMF {
  // expected-warning@-1 {{'NoEligibleSMF' is explicitly marked as device copyable (sycl::is_device_copyable) but has no eligible copy constructor, move constructor, copy assignment operator, or move assignment operator}}
  NoEligibleSMF() {}
  ~NoEligibleSMF() {}
  NoEligibleSMF(const NoEligibleSMF &) = delete;
  NoEligibleSMF(NoEligibleSMF &&) = delete;
  NoEligibleSMF &operator=(const NoEligibleSMF &) = delete;
  NoEligibleSMF &operator=(NoEligibleSMF &&) = delete;
};
template<>
struct sycl::is_device_copyable<NoEligibleSMF> : std::true_type {};

namespace pedanticsycl {

// Custom sycl_kernel_launch that forwards its arguments directly, preventing
// passing by value and the resulting additional copy + decay/deletion. 
// Although technically incorrect, this version of sycl_kernel_launch allows us
// to pass "objects" without triggering its destructor (that we delete for test
// purposes).
template<typename KNT, typename... Ts>
void sycl_kernel_launch(const char *, Ts &&...) {}

// Custom kernel_single_task that takes a ref instead
template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void kernel_single_task(T &t) {}
// expected-note-re@-1 5{{within parameter 't' of type '{{.*}}' declared here}}

// Used to obtain an T& argument without ever constructing a real T object,
// preventing destructors (that we deleted for test purposes) from triggering.
template <typename T> T &getRef();

void test() {
  kernel_single_task<KN<11>>(getRef<BadDestructorDeleted>());
  // expected-note-re@-1 {{in instantiation of function template specialization 'pedanticsycl::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}

  kernel_single_task<KN<12>>(getRef<BadDestructorPrivate>());
  // expected-note-re@-1 {{in instantiation of function template specialization 'pedanticsycl::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}

  PrivateCopyCtor c1;
  kernel_single_task<KN<13>>(c1);
  // expected-note-re@-1 {{in instantiation of function template specialization 'pedanticsycl::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}

  PrivateMoveAssign m1;
  kernel_single_task<KN<14>>(m1);
  // expected-note-re@-1 {{in instantiation of function template specialization 'pedanticsycl::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}

  kernel_single_task<KN<15>>(getRef<NoEligibleSMF>());
  // expected-note-re@-1 {{in instantiation of function template specialization 'pedanticsycl::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
}

} // namespace pedanticsycl
#endif // CHECK_PEDANTIC_SYCL

// Same as previous pendanticsycl testcase, but this time checking warnings are
// not thrown if -Wpendantic-sycl is not enabled
#ifndef CHECK_PEDANTIC_SYCL

struct BadDestructorDeleted {
  // Classes with deleted destructors are still is_trivially_copyable: This
  // copy constructor is added to make the class not trivially copyable.
  BadDestructorDeleted(const BadDestructorDeleted &) {}
  ~BadDestructorDeleted() = delete;
  // expected-warning@-1 {{'BadDestructorDeleted' is explicitly marked as device copyable (sycl::is_device_copyable) but does not have a public, non-deleted destructor}}
};
template<>
struct sycl::is_device_copyable<BadDestructorDeleted> : std::true_type {};

struct PrivateCopyCtor {
  PrivateCopyCtor() {}
private:
  PrivateCopyCtor(const PrivateCopyCtor &) {}
};
template<>
struct sycl::is_device_copyable<PrivateCopyCtor> : std::true_type {};

namespace pedanticsycl {

template<typename KNT, typename... Ts>
void sycl_kernel_launch(const char *, Ts &&...) {}

template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void kernel_single_task(T &t) {}
// expected-note-re@-1 {{within parameter 't' of type '{{.*}}' declared here}}

template <typename T> T &getRef();

void test() {
  // Destructor tests are cheap and still enabled without -Wpendantic-sycl:
  kernel_single_task<KN<16>>(getRef<BadDestructorDeleted>());
  // expected-note-re@-1 {{in instantiation of function template specialization 'pedanticsycl::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}

  // Eligible special member function tests are expensive, and thus should not
  // trigger without -Wpendantic-sycl:
  PrivateCopyCtor c2;
  kernel_single_task<KN<17>>(c2);
}

} // namespace pedanticsycl

#endif // !CHECK_PEDANTIC_SYCL