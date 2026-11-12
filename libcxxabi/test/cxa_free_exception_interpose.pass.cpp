// UNSUPPORTED: target={{.*-apple-darwin.*}}
// RUN: %{cxx} %{flags} %{compile_flags} -O3 %s %{link_flags} -o %t.exe
// RUN: %{exec} %t.exe

#include <cassert>
#include <cstddef>

static bool freed = false;

extern "C" void __cxa_free_exception(void*) {
    freed = true;
}

extern "C" void* __cxa_allocate_exception(std::size_t);

extern "C" void __cxa_decrement_exception_refcount(void*);

int main() {
    void* p = __cxa_allocate_exception(sizeof(int));

    assert(p != nullptr);

    __cxa_decrement_exception_refcount(p);

    assert(freed);
}