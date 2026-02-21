// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-pointer-to-span %t

// Positive: basic (pointer, size) pair.
void process(int *Data, int Size);
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: pointer and size parameters can be replaced with 'std::span'

// Positive: const pointer with unsigned long length.
void readBuf(const char *Buf, unsigned long Len);
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: pointer and size parameters can be replaced with 'std::span'

// Positive: size parameter named "count".
void fill(float *Arr, int Count);
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: pointer and size parameters can be replaced with 'std::span'

// Positive: size parameter named "n".
void copy(int *Dst, int N);
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: pointer and size parameters can be replaced with 'std::span'

// Positive: ptr+size in the middle of more params.
void multi(int Id, double *Data, unsigned Len, bool Flag);
// CHECK-MESSAGES: :[[@LINE-1]]:28: warning: pointer and size parameters can be replaced with 'std::span'

// Negative: second param name does not suggest a size.
void noMatch(int *Data, int Flags);

// Negative: void pointer.
void voidPtr(void *Data, int Size);

// Negative: function pointer.
void funcPtr(void (*Fn)(int), int Size);

// Negative: second param is not an integer.
void wrongType(int *Data, double Size);

// Negative: virtual method.
struct Base {
  virtual void vmethod(int *Data, int Size);
};

// Negative: unnamed size parameter (cannot verify intent).
void unnamed(int *Data, int);
