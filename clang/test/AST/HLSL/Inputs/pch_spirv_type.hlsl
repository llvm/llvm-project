
float2 foo(float2 a, float2 b) {
  return a + b;
}

vk::SpirvOpaqueType</* OpTypeArray */ 28, RWBuffer<float>, vk::integral_constant<uint, 4>> buffers;
