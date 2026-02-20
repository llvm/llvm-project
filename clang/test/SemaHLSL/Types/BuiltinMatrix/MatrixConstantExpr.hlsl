// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -std=hlsl202x -fmatrix-memory-layout=column-major -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -std=hlsl202x -fmatrix-memory-layout=row-major -verify %s

// expected-no-diagnostics

// Matrix subscripting is not currently supported with matrix constexpr. So all
// tests involve casting to another type to determine if the output is correct.

export void fn() {

  // Matrix truncation to int - should get element at (0,0)
  {
    constexpr int2x3 IM = {1, 2, 3,
                           4, 5, 6};
    _Static_assert((int)IM == 1, "Woo!");
  }

  // Matrix splat to vector
  {
    constexpr bool2x2 BM2x2 = true;
    constexpr bool4 BV4 = (bool4)BM2x2;
    _Static_assert(BV4.x == true, "Woo!");
    _Static_assert(BV4.y == true, "Woo!");
    _Static_assert(BV4.z == true, "Woo!");
    _Static_assert(BV4.w == true, "Woo!");
  }

  // Matrix cast to vector
  {
    constexpr float2x2 FM2x2 = {1.5, 2.5, 3.5, 4.5};
    constexpr float4 FV4 = (float4)FM2x2;
    _Static_assert(FV4.x == 1.5, "Woo!");
    _Static_assert(FV4.y == 2.5, "Woo!");
    _Static_assert(FV4.z == 3.5, "Woo!");
    _Static_assert(FV4.w == 4.5, "Woo!");
  }

  // Matrix cast to array
  {
    constexpr float2x2 FM2x2 = {1.5, 2.5, 3.5, 4.5};
    constexpr float FA4[4] = (float[4])FM2x2;
    _Static_assert(FA4[0] == 1.5, "Woo!");
    _Static_assert(FA4[1] == 2.5, "Woo!");
    _Static_assert(FA4[2] == 3.5, "Woo!");
    _Static_assert(FA4[3] == 4.5, "Woo!");
  }

  // Array cast to matrix to vector
  {
    constexpr int IA4[4] = {1, 2, 3, 4};
    constexpr int2x2 IM2x2 = (int2x2)IA4;
    constexpr int4 IV4 = (int4)IM2x2;
    _Static_assert(IV4.x == 1, "Woo!");
    _Static_assert(IV4.y == 2, "Woo!");
    _Static_assert(IV4.z == 3, "Woo!");
    _Static_assert(IV4.w == 4, "Woo!");
  }

  // Vector cast to matrix to vector
  {
    constexpr bool4 BV4_0 = {true, false, true, false};
    constexpr bool2x2 BM2x2 = (bool2x2)BV4_0;
    constexpr bool4 BV4 = (bool4)BM2x2;
    _Static_assert(BV4.x == true, "Woo!");
    _Static_assert(BV4.y == false, "Woo!");
    _Static_assert(BV4.z == true, "Woo!");
    _Static_assert(BV4.w == false, "Woo!");
  }

  // Matrix truncation to vector
  {
    constexpr int3x2 IM3x2 = { 1,  2,
                               3,  4,
                               5,  6};
    constexpr int4 IV4 = (int4)IM3x2;
    _Static_assert(IV4.x == 1, "Woo!");
    _Static_assert(IV4.y == 2, "Woo!");
    _Static_assert(IV4.z == 3, "Woo!");
    _Static_assert(IV4.w == 4, "Woo!");
  }

  // Matrix truncation to array
  {
    constexpr int3x2 IM3x2 = { 1,  2,
                               3,  4,
                               5,  6};
    constexpr int IA4[4] = (int[4])IM3x2;
    _Static_assert(IA4[0] == 1, "Woo!");
    _Static_assert(IA4[1] == 2, "Woo!");
    _Static_assert(IA4[2] == 3, "Woo!");
    _Static_assert(IA4[3] == 4, "Woo!");
  }

  // Array cast to matrix truncation to vector
  {
    constexpr float FA16[16] = { 1.0,  2.0,  3.0,  4.0,
                                 5.0,  6.0,  7.0,  8.0,
                                 9.0, 10.0, 11.0, 12.0,
                                13.0, 14.0, 15.0, 16.0};
    constexpr float4x4 FM4x4 = (float4x4)FA16;
    constexpr float4 FV4 = (float4)FM4x4;
    _Static_assert(FV4.x == 1.0, "Woo!");
    _Static_assert(FV4.y == 2.0, "Woo!");
    _Static_assert(FV4.z == 3.0, "Woo!");
    _Static_assert(FV4.w == 4.0, "Woo!");
  }

  // Vector cast to matrix truncation to vector
  {
    constexpr bool4 BV4 = {true, false, true, false};
    constexpr bool2x2 BM2x2 = (bool2x2)BV4;
    constexpr bool3 BV3 = (bool3)BM2x2;
    _Static_assert(BV4.x == true, "Woo!");
    _Static_assert(BV4.y == false, "Woo!");
    _Static_assert(BV4.z == true, "Woo!");
  }

  // Matrix cast to vector with type conversion
  {
    constexpr float2x2 FM2x2 = {1.5, 2.5, 3.5, 4.5};
    constexpr int4 IV4 = (int4)FM2x2;
    _Static_assert(IV4.x == 1, "Woo!");
    _Static_assert(IV4.y == 2, "Woo!");
    _Static_assert(IV4.z == 3, "Woo!");
    _Static_assert(IV4.w == 4, "Woo!");
  }
}
