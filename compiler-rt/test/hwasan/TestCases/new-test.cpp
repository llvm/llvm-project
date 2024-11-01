// Test basic new functionality.
// RUN: %clangxx_hwasan -std=c++17 %s -o %t -fsized-deallocation
// RUN: %run %t

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <sanitizer/allocator_interface.h>
#include <sanitizer/hwasan_interface.h>

void operator_new_delete(size_t size) {
	void *alloc = operator new(size);
	assert(alloc != nullptr);
	assert(__sanitizer_get_allocated_size(alloc) == size);
	operator delete(alloc);

	alloc = operator new(size);
	assert(alloc != nullptr);
	assert(__sanitizer_get_allocated_size(alloc) == size);
	operator delete(alloc, size);
}

void operator_new_delete_array(size_t size) {
	void *alloc = operator new[](size);
	assert(alloc != nullptr);
	assert(__sanitizer_get_allocated_size(alloc) == size);
	operator delete[](alloc);

	alloc = operator new[](size);
	assert(alloc != nullptr);
	assert(__sanitizer_get_allocated_size(alloc) == size);
	operator delete[](alloc, size);
}

void operator_new_delete(size_t size, std::align_val_t align) {
	void *alloc = operator new(size, align);
	assert(alloc != nullptr);
  assert(reinterpret_cast<uintptr_t>(alloc) % static_cast<uintptr_t>(align) == 0);
	assert(__sanitizer_get_allocated_size(alloc) >= size);
	operator delete(alloc, align);

	alloc = operator new(size, align);
	assert(alloc != nullptr);
  assert(reinterpret_cast<uintptr_t>(alloc) % static_cast<uintptr_t>(align) == 0);
	assert(__sanitizer_get_allocated_size(alloc) >= size);
	operator delete(alloc, size, align);
}

void operator_new_delete_array(size_t size, std::align_val_t align) {
	void *alloc = operator new[](size, align);
	assert(alloc != nullptr);
  assert(reinterpret_cast<uintptr_t>(alloc) % static_cast<uintptr_t>(align) == 0);
	assert(__sanitizer_get_allocated_size(alloc) >= size);
	operator delete[](alloc, align);

	alloc = operator new[](size, align);
	assert(alloc != nullptr);
  assert(reinterpret_cast<uintptr_t>(alloc) % static_cast<uintptr_t>(align) == 0);
	assert(__sanitizer_get_allocated_size(alloc) >= size);
	operator delete[](alloc, size, align);
}

void operator_new_delete(size_t size, const std::nothrow_t &tag) {
	void *alloc = operator new(size, tag);
	assert(alloc != nullptr);
	assert(__sanitizer_get_allocated_size(alloc) == size);
	operator delete(alloc, tag);
}

void operator_new_delete_array(size_t size, const std::nothrow_t &tag) {
	void *alloc = operator new[](size, tag);
	assert(alloc != nullptr);
	assert(__sanitizer_get_allocated_size(alloc) == size);
	operator delete[](alloc, tag);
}

void operator_new_delete(size_t size, std::align_val_t align, const std::nothrow_t &tag) {
	void *alloc = operator new(size, align, tag);
	assert(alloc != nullptr);
  assert(reinterpret_cast<uintptr_t>(alloc) % static_cast<uintptr_t>(align) == 0);
	assert(__sanitizer_get_allocated_size(alloc) >= size);
	operator delete(alloc, align, tag);
}

void operator_new_delete_array(size_t size, std::align_val_t align, const std::nothrow_t &tag) {
	void *alloc = operator new[](size, align, tag);
	assert(alloc != nullptr);
  assert(reinterpret_cast<uintptr_t>(alloc) % static_cast<uintptr_t>(align) == 0);
	assert(__sanitizer_get_allocated_size(alloc) >= size);
	operator delete[](alloc, align, tag);
}

void operator_new_delete(size_t size, void *ptr) {
	void *alloc = operator new(size, ptr);
	assert(alloc == ptr);
	operator delete(alloc, ptr);
}

void operator_new_delete_array(size_t size, void *ptr) {
	void *alloc = operator new[](size, ptr);
	assert(alloc == ptr);
	operator delete[](alloc, ptr);
}

int main() {
  __hwasan_enable_allocator_tagging();

  size_t volatile n = 0;
  char *a1 = new char[n];
  assert(a1 != nullptr);
  assert(__sanitizer_get_allocated_size(a1) == 1);
  delete[] a1;

	constexpr size_t kSize = 8;
	operator_new_delete(kSize);
	operator_new_delete_array(kSize);
	operator_new_delete(kSize, std::nothrow);
	operator_new_delete_array(kSize, std::nothrow);

	char buffer[kSize];
	operator_new_delete(kSize, buffer);
	operator_new_delete_array(kSize, buffer);

#if defined(__cpp_aligned_new) &&                                              \
    (!defined(__GLIBCXX__) ||                                                  \
     (defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE >= 7))
  // Aligned new/delete
  constexpr auto kAlign = std::align_val_t{8};
  void *a2 = ::operator new(4, kAlign);
  assert(a2 != nullptr);
  assert(reinterpret_cast<uintptr_t>(a2) % static_cast<uintptr_t>(kAlign) == 0);
  assert(__sanitizer_get_allocated_size(a2) >= 4);
  ::operator delete(a2, kAlign);

	operator_new_delete(kSize, std::align_val_t{kSize});
	operator_new_delete_array(kSize, std::align_val_t{kSize});
	operator_new_delete(kSize, std::align_val_t{kSize * 2});
	operator_new_delete_array(kSize, std::align_val_t{kSize * 2});
	operator_new_delete(kSize, std::align_val_t{kSize}, std::nothrow);
	operator_new_delete_array(kSize, std::align_val_t{kSize}, std::nothrow);
	operator_new_delete(kSize, std::align_val_t{kSize * 2}, std::nothrow);
	operator_new_delete_array(kSize, std::align_val_t{kSize * 2}, std::nothrow);
#endif
}
