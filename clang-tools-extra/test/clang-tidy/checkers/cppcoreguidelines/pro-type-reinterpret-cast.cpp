// RUN: %check_clang_tidy -check-suffix=DEFAULT             %s cppcoreguidelines-pro-type-reinterpret-cast %t
// RUN: %check_clang_tidy -check-suffix=ALLOW-CAST-TO-BYTES %s cppcoreguidelines-pro-type-reinterpret-cast %t \
// RUN:   -config="{CheckOptions: { \
// RUN:               cppcoreguidelines-pro-type-reinterpret-cast.AllowCastToBytes: True \
// RUN: }}"

int i = 0;
void *j;
void f() { j = reinterpret_cast<void *>(i); }
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:16: warning: do not use reinterpret_cast [cppcoreguidelines-pro-type-reinterpret-cast]
// CHECK-MESSAGES-ALLOW-CAST-TO-BYTES: :[[@LINE-2]]:16: warning: do not use reinterpret_cast [cppcoreguidelines-pro-type-reinterpret-cast]

namespace std
{
enum class byte : unsigned char
{};
}

void check_cast_to_bytes()
{
  float x{};
  auto a = reinterpret_cast<char*>(&x);
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:12: warning: do not use reinterpret_cast [cppcoreguidelines-pro-type-reinterpret-cast]
  auto b = reinterpret_cast<char const*>(&x);
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:12: warning: do not use reinterpret_cast [cppcoreguidelines-pro-type-reinterpret-cast]
  auto c = reinterpret_cast<unsigned char*>(&x);
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:12: warning: do not use reinterpret_cast [cppcoreguidelines-pro-type-reinterpret-cast]
  auto d = reinterpret_cast<unsigned char const*>(&x);
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:12: warning: do not use reinterpret_cast [cppcoreguidelines-pro-type-reinterpret-cast]
  auto e = reinterpret_cast<std::byte*>(&x);
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:12: warning: do not use reinterpret_cast [cppcoreguidelines-pro-type-reinterpret-cast]
  auto f = reinterpret_cast<std::byte const*>(&x);
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:12: warning: do not use reinterpret_cast [cppcoreguidelines-pro-type-reinterpret-cast]

  using CharPtr = char*;
  auto g = reinterpret_cast<CharPtr>(&x);
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:12: warning: do not use reinterpret_cast [cppcoreguidelines-pro-type-reinterpret-cast]
}
