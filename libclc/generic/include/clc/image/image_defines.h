/* get_image_channel_data_type flags */
#define CLK_SNORM_INT8               0x10D0
#define CLK_SNORM_INT16              0x10D1
#define CLK_UNORM_INT8               0x10D2
#define CLK_UNORM_INT16              0x10D3
#define CLK_UNORM_SHORT_565          0x10D4
#define CLK_UNORM_SHORT_555          0x10D5
#define CLK_UNORM_SHORT_101010       0x10D6
#define CLK_SIGNED_INT8              0x10D7
#define CLK_SIGNED_INT16             0x10D8
#define CLK_SIGNED_INT32             0x10D9
#define CLK_UNSIGNED_INT8            0x10DA
#define CLK_UNSIGNED_INT16           0x10DB
#define CLK_UNSIGNED_INT32           0x10DC
#define CLK_HALF_FLOAT               0x10DD
#define CLK_FLOAT                    0x10DE

/* get_image_channel_order flags */
#define CLK_R                        0x10B0
#define CLK_A                        0x10B1
#define CLK_RG                       0x10B2
#define CLK_RA                       0x10B3
#define CLK_RGB                      0x10B4
#define CLK_RGBA                     0x10B5
#define CLK_BGRA                     0x10B6
#define CLK_ARGB                     0x10B7
#define CLK_INTENSITY                0x10B8
#define CLK_LUMINANCE                0x10B9
#define CLK_Rx                       0x10BA
#define CLK_RGx                      0x10BB
#define CLK_RGBx                     0x10BC

/* sampler normalized coords */
#define CLK_NORMALIZED_COORDS_FALSE  0x0000
#define CLK_NORMALIZED_COORDS_TRUE   0x0001
#define __CLC_NORMALIZED_COORDS_MASK 0x0001

/* sampler addressing mode */
#define CLK_ADDRESS_NONE             0x0000
#define CLK_ADDRESS_CLAMP_TO_EDGE    0x0002
#define CLK_ADDRESS_CLAMP            0x0004
#define CLK_ADDRESS_REPEAT           0x0006
#define CLK_ADDRESS_MIRRORED_REPEAT  0x0008
#define __CLC_ADDRESS_MASK           0x000E

/* sampler filter mode */
#define CLK_FILTER_NEAREST           0x0000
#define CLK_FILTER_LINEAR            0x0010
#define __CLC_FILTER_MASK            0x0010
