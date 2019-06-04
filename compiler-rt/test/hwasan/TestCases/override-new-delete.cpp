// RUN: %clangxx_hwasan %s
#include <stddef.h>
#include <new>

char *__dummy;

void *operator new(size_t size) { return __dummy; }
void *operator new[](size_t size) { return __dummy; }
void *operator new(size_t size, std::nothrow_t const&) noexcept { 
  return __dummy; 
}
void *operator new[](size_t size, std::nothrow_t const&) noexcept { 
  return __dummy; 
}

void operator delete(void *ptr) noexcept {}
void operator delete[](void *ptr) noexcept {}
void operator delete(void *ptr, std::nothrow_t const&) noexcept {}
void operator delete[](void *ptr, std::nothrow_t const&) noexcept {}

int main() {
  return 0;  
}
