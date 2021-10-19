// Build an executable with ASan, then extract the DLLs that it depends on.
// RUN: %clang_cl_asan %s -Fe%t.exe
// RUN: llvm-readobj --coff-imports %t.exe | grep Name: | sed -e 's/ *Name: *//' > %t
//
// Make sure the binary doesn't depend on dbghelp directly.
// RUN: not grep dbghelp.dll %t
//
// Make sure any clang_rt DLLs it depends on don't depend on dbghelp. In the
// static build, there won't be any clang_rt DLLs.
// RUN: not grep cl""ang_rt %t || \
// RUN:   grep cl""ang_rt %t | xargs which | \
// RUN:   xargs llvm-readobj --coff-imports | not grep dbghelp.dll %t

extern "C" int puts(const char *);

int main() {
  puts("main");
}
