// RUN: %clangxx %s -o %t -g && %run %t 2>&1 | FileCheck %s

// REQUIRES: internal_symbolizer

#include <assert.h>
#include <dlfcn.h>
#include <link.h>
#include <sanitizer/hwasan_interface.h>
#include <sanitizer/msan_interface.h>
#include <string.h>

#include <string>
#include <vector>

extern "C" {
bool __sanitizer_symbolize_code(const char *ModuleName, uint64_t ModuleOffset,
                                char *Buffer, int MaxLength,
                                bool SymbolizeInlineFrames);
bool __sanitizer_symbolize_data(const char *ModuleName, uint64_t ModuleOffset,
                                char *Buffer, int MaxLength);
bool __sanitizer_symbolize_frame(const char *ModuleName, uint64_t ModuleOffset,
                                 char *Buffer, int MaxLength);
void __sanitizer_print_stack_trace();
bool __sanitizer_symbolize_demangle(const char *Name, char *Buffer,
                                    int MaxLength);
}

struct ScopedInSymbolizer {
#if defined(__has_feature)
#  if __has_feature(memory_sanitizer)
  ScopedInSymbolizer() { __msan_scoped_disable_interceptor_checks(); }
  ~ScopedInSymbolizer() { __msan_scoped_enable_interceptor_checks(); }
#  endif
#endif
};

struct FrameInfo {
  int line;
  std::string file;
  std::string function;
  void *address;
};

__attribute__((noinline)) void *GetPC() { return __builtin_return_address(0); }

__attribute__((always_inline)) FrameInfo InlineFunction() {
  void *address = GetPC();
  return {0, "", "",
          reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(address) - 1)};
}

__attribute__((noinline)) FrameInfo NoInlineFunction() {
  void *address = GetPC();
  return {0, "", "",
          reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(address) - 1)};
}

template <int N> struct A {
  template <class T> FrameInfo RecursiveTemplateFunction(const T &t);
};

template <int N>
template <class T>
__attribute__((noinline)) FrameInfo A<N>::RecursiveTemplateFunction(const T &) {
  std::vector<T> t;
  return A<N - 1>().RecursiveTemplateFunction(t);
}

template <>
template <class T>
__attribute__((noinline)) FrameInfo A<0>::RecursiveTemplateFunction(const T &) {
  return NoInlineFunction();
}

__attribute__((no_sanitize_memory)) std::pair<const char *, uint64_t>
GetModuleAndOffset(const void *address) {
  Dl_info di;
  link_map *lm = nullptr;
#if __has_feature(hwaddress_sanitizer)
  address = __hwasan_tag_pointer(address, 0);
#endif
  assert(
      dladdr1(address, &di, reinterpret_cast<void **>(&lm), RTLD_DL_LINKMAP));
  return {di.dli_fname, reinterpret_cast<uint64_t>(address) - lm->l_addr};
}

std::string Symbolize(FrameInfo frame) {
  auto modul_offset = GetModuleAndOffset(frame.address);
  char buffer[1024] = {};
  ScopedInSymbolizer in_symbolizer;
  assert(__sanitizer_symbolize_code(modul_offset.first, modul_offset.second,
                                    buffer, std::size(buffer), true));
  return buffer;
}

std::string GetRegex(const FrameInfo &frame) {
  return frame.function + "[^\\n]*\\n[^\\n]*" + frame.file + ":" +
         std::to_string(frame.line);
}

void TestInline() {
  auto frame = InlineFunction();
  fprintf(stderr, "%s: %s\n", __FUNCTION__, Symbolize(frame).c_str());
  // CHECK-LABEL: TestInline: InlineFunction()
  // CHECK-NEXT: internal_symbolizer.cpp:[[# @LINE - 58]]
  // CHECK-NEXT: TestInline()
  // CHECK-NEXT: internal_symbolizer.cpp:[[# @LINE - 5]]
}

void TestNoInline() {
  auto frame = NoInlineFunction();
  fprintf(stderr, "%s: %s\n", __FUNCTION__, Symbolize(frame).c_str());
  // CHECK-LABEL: TestNoInline: NoInlineFunction()
  // CHECK-NEXT: internal_symbolizer.cpp:[[# @LINE - 61]]
}

void TestLongFunctionNames() {
  auto frame = A<10>().RecursiveTemplateFunction(0);
  fprintf(stderr, "%s: %s\n", __FUNCTION__, Symbolize(frame).c_str());
  // CHECK-LABEL: TestLongFunctionNames: NoInlineFunction()
  // CHECK-NEXT: internal_symbolizer.cpp:[[# @LINE - 68]]
}

std::string SymbolizeStaticVar() {
  static int var = 1;
  auto modul_offset = GetModuleAndOffset(&var);
  char buffer[1024] = {};
  ScopedInSymbolizer in_symbolizer;
  assert(__sanitizer_symbolize_data(modul_offset.first, modul_offset.second,
                                    buffer, std::size(buffer)));
  return buffer;
}

void TestData() {
  fprintf(stderr, "%s: %s\n", __FUNCTION__, SymbolizeStaticVar().c_str());
  // CHECK-LABEL: TestData: SymbolizeStaticVar[abi:cxx11]()::var
  // CHECK-NEXT: {{[0-9]+ +[0-9]+}}
  // CHECK-NEXT: internal_symbolizer.cpp:[[# @LINE - 13]]
}

__attribute__((noinline)) std::string SymbolizeLocalVars(const void *pc) {
  auto modul_offset = GetModuleAndOffset(pc);
  char buffer[1024] = {};
  ScopedInSymbolizer in_symbolizer;
  __sanitizer_symbolize_frame(modul_offset.first, modul_offset.second, buffer,
                              std::size(buffer));
  return buffer;
}

__attribute__((
    noinline,
    no_sanitize_address /* Asan merges allocas destroying variable DI */)) void
TestFrame() {
  volatile int var = 1;
  void *address = GetPC();
  fprintf(stderr, "%s: %s\n", __FUNCTION__,
          SymbolizeLocalVars(address).c_str());
  // CHECK-LABEL: TestFrame: TestFrame
  // CHECK-NEXT: var
  // CHECK-NEXT: internal_symbolizer.cpp:[[# @LINE - 6]]
  // CHECK-NEXT: {{-?[0-9]+ +[0-9]+}}
  // CHECK-NEXT: TestFrame
  // CHECK-NEXT: address
  // CHECK-NEXT: internal_symbolizer.cpp:[[# @LINE - 9]]
  // CHECK-NEXT: {{-?[0-9]+ +[0-9]+}}
}

void TestDemangle() {
  char out[128];
  assert(!__sanitizer_symbolize_demangle("1A", out, sizeof(out)));

  const char name[] = "_Z3fooi";
  for (int i = 1; i < sizeof(out); ++i) {
    memset(out, 1, sizeof(out));
    assert(__sanitizer_symbolize_demangle(name, out, i) == (i > 8));
    assert(i < 9 || 0 == strncmp(out, "foo(int)", i - 1));
  }
}

int main() {
  TestInline();
  TestNoInline();
  TestLongFunctionNames();
  TestData();
  TestFrame();
  TestDemangle();
}
