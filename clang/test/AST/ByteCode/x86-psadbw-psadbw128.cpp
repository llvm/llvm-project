// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -O0 \
// RUN:   -fexperimental-new-constant-interpreter -verify %s

// No headers allowed. Use Clang's vector types directly.

typedef char v16qi __attribute__((vector_size(16)));
typedef long long v2di __attribute__((vector_size(16)));

constexpr v2di test_psadbw128() {
  v16qi a = {10,20,30,40,50,60,70,80,
             1,2,3,4,5,6,7,8};

  v16qi b = {5,15,25,45,55,55,75,85,
             10,0,3,9,1,10,2,8};

  // Call the builtin directly
  return __builtin_ia32_psadbw128(a, b);
}

static_assert(test_psadbw128()[0] == 40, "block0 mismatch");
static_assert(test_psadbw128()[1] == 29, "block1 mismatch");
