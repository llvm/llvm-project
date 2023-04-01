// UNSUPPORTED: target={{.*-windows-gnu}}

// Make sure we can link a DLL with large functions which would mean
// functions such as __asan_loadN and __asan_storeN will be called
// from the DLL.  We simulate the large function with
// -mllvm -asan-instrumentation-with-call-threshold=0.
// RUN: %clang_cl_asan %s -c -Fo%t.obj -mllvm -asan-instrumentation-with-call-threshold=0
// RUN: lld-link /nologo /DLL /OUT:%t.dll %t.obj %asan_dll_thunk
// REQUIRES: asan-static-runtime
// REQUIRES: lld-available

void f(long* foo, long* bar) {
  // One load and one store
  *foo = *bar;
}
