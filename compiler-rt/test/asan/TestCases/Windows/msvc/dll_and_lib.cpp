// Just make sure we can link an implib into another DLL
// This used to fail between r212699 and r212814.
// RUN: %clang_cl_asan -DCONFIG=1 %s -c -Fo%t.1.obj
// RUN: lld-link /nologo /DLL /OUT:%t.1.dll %t.1.obj %asan_dll_thunk
// RUN: %clang_cl_asan -DCONFIG=2 %s -c -Fo%t.2.obj
// RUN: lld-link /nologo /DLL /OUT:%t.2.dll %t.2.obj %t.1.lib %asan_dll_thunk
// REQUIRES: asan-static-runtime
// REQUIRES: lld-available

#if CONFIG==1
extern "C" __declspec(dllexport) int f1() {
  int x = 0;
  return 1;
}
#else
extern "C" __declspec(dllexport) int f2() {
  int x = 0;
  return 2;
}
#endif
