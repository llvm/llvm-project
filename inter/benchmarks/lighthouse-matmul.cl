#ifndef MATRIX_SIZE
#define MATRIX_SIZE 128
#endif

#ifndef REDUCTION_SIZE
#define REDUCTION_SIZE 64
#endif

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(256, 1, 1)))
kernel void payload_kernel(global half *a, global half *b, global float *c) {
  uint subgroup = get_sub_group_id();
  uint tileColumn = subgroup % 4;
  uint tileRow = subgroup / 4;
  uint column = get_group_id(1) * 64 + tileColumn * 16;
  uint row = get_group_id(0) * 64 + tileRow * 16;
  float8 acc0 = 0.0f;
  float8 acc1 = 0.0f;

  intel_sub_group_2d_block_prefetch_16b_8r16x1c(
      a, REDUCTION_SIZE * 2, MATRIX_SIZE, REDUCTION_SIZE * 2, (int2)(0, row));
  intel_sub_group_2d_block_prefetch_16b_8r16x1c(
      b, MATRIX_SIZE * 2, REDUCTION_SIZE, MATRIX_SIZE * 2, (int2)(column, 0));

  for (uint inner = 0; inner < REDUCTION_SIZE; inner += 32) {
    ushort8 a00;
    ushort8 a01;
    ushort8 a10;
    ushort8 a11;
    uint8 b0;
    uint8 b1;

#ifndef DROP_LOOP_PREFETCH
    intel_sub_group_2d_block_prefetch_16b_8r16x1c(
        b, MATRIX_SIZE * 2, REDUCTION_SIZE, MATRIX_SIZE * 2,
        (int2)(column, inner + 32));
#endif
    intel_sub_group_2d_block_read_16b_8r16x1c(
        a, REDUCTION_SIZE * 2, MATRIX_SIZE, REDUCTION_SIZE * 2,
        (int2)(inner, row),
        (private ushort *)&a00);
    intel_sub_group_2d_block_read_16b_8r16x1c(
        a, REDUCTION_SIZE * 2, MATRIX_SIZE, REDUCTION_SIZE * 2,
        (int2)(inner + 16, row),
        (private ushort *)&a01);
    intel_sub_group_2d_block_read_16b_8r16x1c(
        a, REDUCTION_SIZE * 2, MATRIX_SIZE, REDUCTION_SIZE * 2,
        (int2)(inner, row + 8),
        (private ushort *)&a10);
    intel_sub_group_2d_block_read_16b_8r16x1c(
        a, REDUCTION_SIZE * 2, MATRIX_SIZE, REDUCTION_SIZE * 2,
        (int2)(inner + 16, row + 8),
        (private ushort *)&a11);
    intel_sub_group_2d_block_read_transform_16b_16r16x1c(
        b, MATRIX_SIZE * 2, REDUCTION_SIZE, MATRIX_SIZE * 2,
        (int2)(column, inner),
        (private uint *)&b0);
    intel_sub_group_2d_block_read_transform_16b_16r16x1c(
        b, MATRIX_SIZE * 2, REDUCTION_SIZE, MATRIX_SIZE * 2,
        (int2)(column, inner + 16), (private uint *)&b1);
    acc0 = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(a00), as_int8(b0),
                                                   acc0);
    acc0 = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(a01), as_int8(b1),
                                                   acc0);
    acc1 = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(a10), as_int8(b0),
                                                   acc1);
    acc1 = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(a11), as_int8(b1),
                                                   acc1);
  }

  uint8 bits0 = as_uint8(acc0);
  uint8 bits1 = as_uint8(acc1);
  intel_sub_group_2d_block_write_32b_8r16x1c(
      c, MATRIX_SIZE * 4, MATRIX_SIZE, MATRIX_SIZE * 4, (int2)(column, row),
      (private uint *)&bits0);
  intel_sub_group_2d_block_write_32b_8r16x1c(
      c, MATRIX_SIZE * 4, MATRIX_SIZE, MATRIX_SIZE * 4,
      (int2)(column, row + 8), (private uint *)&bits1);
}
