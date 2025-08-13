//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// ADDITIONAL_COMPILE_FLAGS: -O0 -g -gdwarf-3
// FIXME: also requires _LIBCPP_STACKTRACE_ALLOW_TOOLS_AT_RUNTIME

/*
Note: requires _LIBCPP_STACKTRACE_ALLOW_TOOLS_AT_RUNTIME, as well as at least
one such tool installed on the local system and findable with PATH.
You can also run this locally like so:

```
BUILDDIR=build
ninja -C "${BUILDDIR}" cxx-test-depends

# Build and run via lit.  Use the default program names and let `env` try to find full paths.
"${BUILDDIR}/bin/llvm-lit" -sv libcxx/test/libcxx/stacktrace/use_available_progs.pass.cpp

# To force use of a particular path for a tool (not relying on PATH), specify these env variables
and run the test program directly (`lit` won't pass these variables).  Use `/bin/false` to disable a tool.
Examples:

LIBCXX_STACKTRACE_FORCE_GNU_ADDR2LINE_PATH=/opt/homebrew/Cellar/binutils/2.45/bin/addr2line \
LIBCXX_STACKTRACE_FORCE_APPLE_ATOS_PATH=/usr/bin/atos \
LIBCXX_STACKTRACE_FORCE_LLVM_SYMBOLIZER_PATH=/opt/homebrew/Cellar/llvm/20.1.7/bin/llvm-symbolizer \
  $BUILDDIR/libcxx/test/libcxx/stacktrace/Output/use_available_progs.pass.cpp.dir/t.tmp.exe
```
*/

#include <__config>
#include <__stacktrace/memory.h>
#include <cassert>
#include <iostream>
#include <stacktrace>

#include <__stacktrace/images.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct addr2line;
struct arena;
struct atos;
struct base;
struct llvm_symbolizer;

template <class T>
struct __executable_name {
  _LIBCPP_EXPORTED_FROM_ABI static char const* get();
};

template <class T>
_LIBCPP_EXPORTED_FROM_ABI bool __has_working_executable();

template <class T>
bool __run_tool(base&, arena&);

extern template struct __executable_name<addr2line>;
extern template bool __has_working_executable<addr2line>();
extern template bool __run_tool<addr2line>(base&, arena&);

extern template struct __executable_name<atos>;
extern template bool __has_working_executable<atos>();
extern template bool __run_tool<atos>(base&, arena&);

extern template struct __executable_name<llvm_symbolizer>;
extern template bool __has_working_executable<llvm_symbolizer>();
extern template bool __run_tool<llvm_symbolizer>(base&, arena&);

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

int func0() { return 1000; }
constexpr static int _FUNC0_LINE = __LINE__ - 1;

int func1() { return 1001; }
constexpr static int _FUNC1_LINE = __LINE__ - 1;

int func2() { return 1002; }
constexpr static int _FUNC2_LINE = __LINE__ - 1;

namespace {

/**
Produce a fake stacktrace with 3 entries, having (in order of index) addresses of `func0`, `func1`, `func2`.
Only addresses are populated; no symbols / source locations are in these entries.
The addresses are the base addrs of these functions (not instructions' addresses, nor
are they decremented by 1 to get the calling instruction and not return address, as the unwinder would).
These addresses come from the main program's address space (this one, "use_available_progs.pass.cpp")
so populate their `__image_` with a pointer to the main program image, so address adjustment
(for ASLR) works.
*/
std::stacktrace fake_stacktrace() {
  static std::__stacktrace::images imgs;
  static auto* main_image = imgs.main_prog_image();

  std::stacktrace ret;
  auto& base  = *(std::__stacktrace::base*)(&ret);
  auto& e0    = base.__emplace_entry_();
  e0.__addr_  = uintptr_t(&func0);
  e0.__image_ = main_image;
  auto& e1    = base.__emplace_entry_();
  e1.__addr_  = uintptr_t(&func1);
  e1.__image_ = main_image;
  auto& e2    = base.__emplace_entry_();
  e2.__addr_  = uintptr_t(&func2);
  e2.__image_ = main_image;
  return ret;
}

void check_stacktrace(std::stacktrace& st) {
  assert(st.at(0).native_handle() == uintptr_t(&func0));
  assert(st.at(0).description().contains("func0")); // e.g.: _func0, func0, func0(), other variations maybe
  assert(st.at(0).source_file().ends_with("use_available_progs.pass.cpp"));
  assert(st.at(0).source_line() == _FUNC0_LINE);

  assert(st.at(1).native_handle() == uintptr_t(&func1));
  assert(st.at(1).description().contains("func1"));
  assert(st.at(1).source_file().ends_with("use_available_progs.pass.cpp"));
  assert(st.at(1).source_line() == _FUNC1_LINE);

  assert(st.at(2).native_handle() == uintptr_t(&func2));
  assert(st.at(2).description().contains("func2"));
  assert(st.at(2).source_file().ends_with("use_available_progs.pass.cpp"));
  assert(st.at(2).source_line() == _FUNC2_LINE);
}

template <class T>
int try_tool() {
  std::cerr << "*** trying tool: " << std::__stacktrace::__executable_name<T>::get() << '\n';
  if (std::__stacktrace::__has_working_executable<T>()) {
    auto st    = fake_stacktrace();
    auto& base = (std::__stacktrace::base&)st;
    std::__stacktrace::stack_bytes<std::__stacktrace::base::__k_init_pool_on_stack> stack_bytes;
    std::__stacktrace::byte_pool stack_pool = stack_bytes.pool();
    std::__stacktrace::arena arena(stack_pool, st.get_allocator());
    std::__stacktrace::__run_tool<T>(base, arena);
    std::cout << st << std::endl;
    check_stacktrace(st);
    std::cerr << "... succeeded\n";
    return 1;
  } else {
    std::cerr << "... not found\n";
  }
  return 0;
}

} // namespace

int main(int, char**) {
  /*
  If for some reason all tools failed to run, we don't quite want to declare a success,
  so this is false until a tool ran (and succeeded).

  If any of these tools exist, but the stacktrace operation failed when using it,
  the `assert`s within that test will abort immediately.

  Therefore, we can't assume one's machine (or CI) has any one of these tools; but assume
  it will have at least _something_, and ensure that something works.
  */
  int something_worked = 0;
  something_worked += try_tool<std::__stacktrace::addr2line>();
  something_worked += try_tool<std::__stacktrace::atos>();
  something_worked += try_tool<std::__stacktrace::llvm_symbolizer>();
  assert(something_worked);
  return 0;
}
