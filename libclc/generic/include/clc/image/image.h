_CLC_OVERLOAD _CLC_DECL int get_image_width (image2d_t image);
_CLC_OVERLOAD _CLC_DECL int get_image_width (image3d_t image);

_CLC_OVERLOAD _CLC_DECL int get_image_height (image2d_t image);
_CLC_OVERLOAD _CLC_DECL int get_image_height (image3d_t image);

_CLC_OVERLOAD _CLC_DECL int get_image_depth (image3d_t image);

_CLC_OVERLOAD _CLC_DECL int get_image_channel_data_type (image2d_t image);
_CLC_OVERLOAD _CLC_DECL int get_image_channel_data_type (image3d_t image);

_CLC_OVERLOAD _CLC_DECL int get_image_channel_order (image2d_t image);
_CLC_OVERLOAD _CLC_DECL int get_image_channel_order (image3d_t image);

_CLC_OVERLOAD _CLC_DECL int2 get_image_dim (image2d_t image);
_CLC_OVERLOAD _CLC_DECL int4 get_image_dim (image3d_t image);

_CLC_OVERLOAD _CLC_DECL void
write_imagef(image2d_t image, int2 coord, float4 color);
_CLC_OVERLOAD _CLC_DECL void
write_imagei(image2d_t image, int2 coord, int4 color);
_CLC_OVERLOAD _CLC_DECL void
write_imageui(image2d_t image, int2 coord, uint4 color);

_CLC_OVERLOAD _CLC_DECL float4
read_imagef(image2d_t image, sampler_t sampler, int2 coord);
_CLC_OVERLOAD _CLC_DECL float4
read_imagef(image2d_t image, sampler_t sampler, float2 coord);
_CLC_OVERLOAD _CLC_DECL int4
read_imagei(image2d_t image, sampler_t sampler, int2 coord);
_CLC_OVERLOAD _CLC_DECL int4
read_imagei(image2d_t image, sampler_t sampler, float2 coord);
_CLC_OVERLOAD _CLC_DECL uint4
read_imageui(image2d_t image, sampler_t sampler, int2 coord);
_CLC_OVERLOAD _CLC_DECL uint4
read_imageui(image2d_t image, sampler_t sampler, float2 coord);
