// RUN: %clangxx_dfsan %s -mllvm -dfsan-fast-16-labels -mllvm -dfsan-track-select-control-flow=false -mllvm -dfsan-combine-pointer-labels-on-load=false -O0 -DO0 -o %t && %run %t
// RUN: %clangxx_dfsan %s -mllvm -dfsan-fast-16-labels -mllvm -dfsan-track-select-control-flow=false -mllvm -dfsan-combine-pointer-labels-on-load=false -O1 -o %t && %run %t
//
// REQUIRES: x86_64-target-arch

#include <algorithm>
#include <assert.h>
#include <sanitizer/dfsan_interface.h>
#include <utility>

__attribute__((noinline))
std::pair<int *, int>
make_pair(int *p, int i) { return {p, i}; }

__attribute__((noinline))
std::pair<int *, int>
copy_pair1(const std::pair<int *, int> &pair) {
  return pair;
}

__attribute__((noinline))
std::pair<int *, int>
copy_pair2(std::pair<int *, int> *pair) {
  return *pair;
}

__attribute__((noinline))
std::pair<int *, int>
copy_pair3(std::pair<int *, int> &&pair) {
  return std::move(pair);
}

__attribute__((noinline))
std::pair<const char *, uint32_t>
return_ptr_and_i32(const char *p, uint32_t res) {
  for (uint32_t i = 2; i < 5; i++) {
    uint32_t byte = static_cast<uint8_t>(p[i]);
    res += (byte - 1) << (7 * i);
    if (byte < 128) {
      return {p + i + 1, res};
    }
  }
  return {nullptr, 0};
}

__attribute__((noinline))
std::pair<const char *, uint64_t>
return_ptr_and_i64(const char *p, uint32_t res32) {
  uint64_t res = res32;
  for (uint32_t i = 2; i < 10; i++) {
    uint64_t byte = static_cast<uint8_t>(p[i]);
    res += (byte - 1) << (7 * i);
    if (byte < 128) {
      return {p + i + 1, res};
    }
  }
  return {nullptr, 0};
}

void test_simple_constructors() {
  int i = 1;
  int *ptr = NULL;
  dfsan_set_label(8, &i, sizeof(i));
  dfsan_set_label(2, &ptr, sizeof(ptr));

  std::pair<int *, int> pair1 = make_pair(ptr, i);
  int i1 = pair1.second;
  int *ptr1 = pair1.first;

#ifdef O0
  assert(dfsan_read_label(&i1, sizeof(i1)) == 10);
  assert(dfsan_read_label(&ptr1, sizeof(ptr1)) == 10);
#else
  assert(dfsan_read_label(&i1, sizeof(i1)) == 8);
  assert(dfsan_read_label(&ptr1, sizeof(ptr1)) == 2);
#endif

  std::pair<int *, int> pair2 = copy_pair1(pair1);
  int i2 = pair2.second;
  int *ptr2 = pair2.first;

#ifdef O0
  assert(dfsan_read_label(&i2, sizeof(i2)) == 10);
  assert(dfsan_read_label(&ptr2, sizeof(ptr2)) == 10);
#else
  assert(dfsan_read_label(&i2, sizeof(i2)) == 8);
  assert(dfsan_read_label(&ptr2, sizeof(ptr2)) == 2);
#endif

  std::pair<int *, int> pair3 = copy_pair2(&pair1);
  int i3 = pair3.second;
  int *ptr3 = pair3.first;

#ifdef O0
  assert(dfsan_read_label(&i3, sizeof(i3)) == 10);
  assert(dfsan_read_label(&ptr3, sizeof(ptr3)) == 10);
#else
  assert(dfsan_read_label(&i3, sizeof(i3)) == 8);
  assert(dfsan_read_label(&ptr3, sizeof(ptr3)) == 2);
#endif

  std::pair<int *, int> pair4 = copy_pair3(std::move(pair1));
  int i4 = pair4.second;
  int *ptr4 = pair4.first;

#ifdef O0
  assert(dfsan_read_label(&i4, sizeof(i4)) == 10);
  assert(dfsan_read_label(&ptr4, sizeof(ptr4)) == 10);
#else
  assert(dfsan_read_label(&i4, sizeof(i4)) == 8);
  assert(dfsan_read_label(&ptr4, sizeof(ptr4)) == 2);
#endif
}

void test_branches() {
  uint32_t res = 4;
  dfsan_set_label(8, &res, sizeof(res));

  char p[100];
  const char *q = p;
  dfsan_set_label(2, &q, sizeof(q));

  {
    std::fill_n(p, 100, static_cast<char>(128));

    {
      std::pair<const char *, uint32_t> r = return_ptr_and_i32(q, res);
      assert(dfsan_read_label(&r.first, sizeof(r.first)) == 0);
      assert(dfsan_read_label(&r.second, sizeof(r.second)) == 0);
    }

    {
      std::pair<const char *, uint64_t> r = return_ptr_and_i64(q, res);
      assert(dfsan_read_label(&r.first, sizeof(r.first)) == 0);
      assert(dfsan_read_label(&r.second, sizeof(r.second)) == 0);
    }
  }

  {
    std::fill_n(p, 100, 0);

    {
      std::pair<const char *, uint32_t> r = return_ptr_and_i32(q, res);
#ifdef O0
      assert(dfsan_read_label(&r.first, sizeof(r.first)) == 10);
      assert(dfsan_read_label(&r.second, sizeof(r.second)) == 10);
#else
      assert(dfsan_read_label(&r.first, sizeof(r.first)) == 2);
      assert(dfsan_read_label(&r.second, sizeof(r.second)) == 8);
#endif
    }

    {
      std::pair<const char *, uint64_t> r = return_ptr_and_i64(q, res);
#ifdef O0
      assert(dfsan_read_label(&r.first, sizeof(r.first)) == 10);
      assert(dfsan_read_label(&r.second, sizeof(r.second)) == 10);
#else
      assert(dfsan_read_label(&r.first, sizeof(r.first)) == 2);
      assert(dfsan_read_label(&r.second, sizeof(r.second)) == 8);
#endif
    }
  }
}

int main(void) {
  test_simple_constructors();
  test_branches();

  return 0;
}
