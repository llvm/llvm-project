// RUN: %check_clang_tidy %s misc-misplaced-const %t

typedef int plain_i;
typedef int *ip;
typedef const int *cip;

typedef void (*func_ptr)(void);

void func(void) {
  // ok
  const int *i0 = 0;
  const plain_i *i1 = 0;
  const cip i2 = 0; // const applies to both pointer and pointee.

  // Not ok
  const ip i3 = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: 'i3' declared with a const-qualified typedef; results in the type being 'int *const' instead of 'const int *'

  ip const i4 = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: 'i4' declared with a const-qualified typedef; results in the type being 'int *const' instead of 'const int *'

  const volatile ip i5 = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: 'i5' declared with a const-qualified typedef; results in the type being 'int *const volatile' instead of 'const int *volatile'
}

void func2(const plain_i *i1,
           const cip i2,
           const ip i3,
           // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: 'i3' declared with a const-qualified
           const int *i4) {
}

struct S {
  const int *i0;
  const plain_i *i1;
  const cip i2;
  const ip i3;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: 'i3' declared with a const-qualified
};

// Function pointers should not be diagnosed because a function
// pointer type can never be const.
void func3(const func_ptr fp) {
  const func_ptr fp2 = fp;
}
