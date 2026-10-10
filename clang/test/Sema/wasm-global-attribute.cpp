// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple wasm64-unknown-unknown-wasm -std=c++17 -fsyntax-only -verify %s

int wasm_global_scalar [[clang::wasm_global]];
int wasm_global_second [[clang::wasm_global]];
int *wasm_global_pointer [[clang::wasm_global]];
bool wasm_global_bool [[clang::wasm_global]];
long long wasm_global_integer [[clang::wasm_global]];
unsigned _BitInt(64) wasm_global_bitint [[clang::wasm_global]];
float wasm_global_float [[clang::wasm_global]];
double wasm_global_double [[clang::wasm_global]];
enum class ScalarEnum { Zero };
ScalarEnum wasm_global_enum [[clang::wasm_global]];
thread_local int wasm_global_tls [[clang::wasm_global]]; // expected-error {{'wasm_global' attribute cannot be applied to a thread-local variable}}
__thread int wasm_global_gnu_tls [[clang::wasm_global]]; // expected-error {{'wasm_global' attribute cannot be applied to a thread-local variable}}

struct Record { int value; };
using MemberFunction = void (Record::*)();
using MemberData = int Record::*;
_Complex double wasm_global_complex [[clang::wasm_global]]; // expected-error {{'wasm_global' attribute does not support type}}
MemberFunction wasm_global_member_function [[clang::wasm_global]]; // expected-error {{'wasm_global' attribute does not support type}}
MemberData wasm_global_member_data [[clang::wasm_global]]; // expected-error {{'wasm_global' attribute does not support type}}
long double wasm_global_long_double [[clang::wasm_global]]; // expected-error {{'wasm_global' attribute does not support type}}
unsigned _BitInt(65) wasm_global_wide_bitint [[clang::wasm_global]]; // expected-error {{'wasm_global' attribute does not support type}}
__int128 wasm_global_int128 [[clang::wasm_global]]; // expected-error {{'wasm_global' attribute does not support type}}

auto wasm_global_auto [[clang::wasm_global]] = 0;
auto wasm_global_auto_pointer [[clang::wasm_global]] = static_cast<int *>(nullptr);
decltype(auto) wasm_global_decltype_auto [[clang::wasm_global]] = 0;
auto wasm_global_auto_record [[clang::wasm_global]] = Record{}; // expected-error {{'wasm_global' attribute requires a scalar type}}
auto wasm_global_auto_wide [[clang::wasm_global]] = (__int128)0; // expected-error {{'wasm_global' attribute does not support type}}

template <class T>
T wasm_global_template [[clang::wasm_global]] = T{}; // expected-error {{'wasm_global' attribute requires a scalar type}} expected-error {{'wasm_global' attribute does not support type}}
int read_wasm_global_template() { return wasm_global_template<int>; }
Record instantiate_record = wasm_global_template<Record>; // expected-note {{in instantiation of variable template specialization}}
auto instantiate_wide = wasm_global_template<__int128>; // expected-note {{in instantiation of variable template specialization}}

template <class T>
struct ClassGlobal {
  static T value [[clang::wasm_global]]; // expected-error {{'wasm_global' attribute requires a scalar type}}
};
int read_class_global() { return ClassGlobal<int>::value; }
Record instantiate_class_record = ClassGlobal<Record>::value; // expected-note {{in instantiation of template class}}

using WasmInt = int __attribute__((address_space(1)));

void wasm_global_address_test(bool condition) {
  auto pointer = &wasm_global_scalar; // expected-error {{cannot take the address of a WebAssembly global}}
  auto builtin_pointer = __builtin_addressof(wasm_global_scalar); // expected-error {{cannot take the address of a WebAssembly global}}
  auto comma_pointer = &((void)0, wasm_global_scalar); // expected-error {{cannot take the address of a WebAssembly global}}
  auto conditional_pointer = &(condition ? wasm_global_scalar : wasm_global_second); // expected-error {{cannot take the address of a WebAssembly global}}
  auto conditional_builtin_pointer = __builtin_addressof(condition ? wasm_global_scalar : wasm_global_second); // expected-error {{cannot take the address of a WebAssembly global}}
  auto assignment_pointer = &(wasm_global_scalar = 0); // expected-error {{cannot take the address of a WebAssembly global}}
  auto increment_pointer = &++wasm_global_scalar; // expected-error {{cannot take the address of a WebAssembly global}}
  WasmInt &reference = wasm_global_scalar; // expected-error {{cannot bind a reference to a WebAssembly global}}
  WasmInt &comma_reference = ((void)0, wasm_global_scalar); // expected-error {{cannot bind a reference to a WebAssembly global}}
  WasmInt &conditional_reference = condition ? wasm_global_scalar : wasm_global_second; // expected-error {{cannot bind a reference to a WebAssembly global}}
  WasmInt &assignment_reference = (wasm_global_scalar = 0); // expected-error {{cannot bind a reference to a WebAssembly global}}
  WasmInt &increment_reference = ++wasm_global_scalar; // expected-error {{cannot bind a reference to a WebAssembly global}}
  auto &&xvalue_reference = static_cast<WasmInt &&>(wasm_global_scalar); // expected-error {{cannot bind a reference to a WebAssembly global}}

  const int &copied_reference = static_cast<int>(wasm_global_scalar);
  const double &converted_reference = static_cast<double>(wasm_global_scalar);
  auto &&value_reference = +wasm_global_scalar;
  auto copied_pointer_value = wasm_global_pointer;
  auto pointee_pointer = &*wasm_global_pointer;
  auto &pointee_reference = *wasm_global_pointer;
  int ordinary;
  auto ordinary_pointer = &ordinary;
  auto &ordinary_reference = ordinary;
}