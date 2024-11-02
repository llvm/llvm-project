// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc -verify %s
//

typedef __SIZE_TYPE__ size_t;

typedef enum memory_order {
  memory_order_relaxed = __ATOMIC_RELAXED,
} memory_order;

void *calloc(size_t, size_t);
void free(void *);

struct SomeData {
  int i;
  _Atomic int ref;
};

static struct SomeData *alloc_data(void)
{
  struct SomeData *data = calloc(sizeof(*data), 1);

  __c11_atomic_store(&data->ref, 2, memory_order_relaxed);
  return data;
}

static void put_data(struct SomeData *data)
{
 if (__c11_atomic_fetch_sub(&data->ref, 1, memory_order_relaxed) == 1)
   free(data);
}

static int dec_refcounter(struct SomeData *data)
{
  return __c11_atomic_fetch_sub(&data->ref, 1, memory_order_relaxed) == 1;
}

static void put_data_nested(struct SomeData *data)
{
  if (dec_refcounter(data))
    free(data);
}

static void put_data_uncond(struct SomeData *data)
{
  free(data);
}

static void put_data_unrelated_atomic(struct SomeData *data)
{
  free(data);
  __c11_atomic_fetch_sub(&data->ref, 1, memory_order_relaxed);
}

void test_no_uaf(void)
{
  struct SomeData *data = alloc_data();
  put_data(data);
  data->i += 1; // no warning
}

void test_no_uaf_nested(void)
{
  struct SomeData *data = alloc_data();
  put_data_nested(data);
  data->i += 1; // no warning
}

void test_uaf(void)
{
  struct SomeData *data = alloc_data();
  put_data_uncond(data);
  data->i += 1; // expected-warning{{Use of memory after it is freed}}
}

void test_no_uaf_atomic_after(void)
{
  struct SomeData *data = alloc_data();
  put_data_unrelated_atomic(data);
  data->i += 1; // expected-warning{{Use of memory after it is freed}}
}
