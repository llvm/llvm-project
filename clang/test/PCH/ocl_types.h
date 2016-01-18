/* Used with the ocl_types.cl test */

// image1d_t
typedef image1d_t img1d_t;

// image1d_array_t
typedef image1d_array_t img1darr_t;

// image1d_buffer_t
typedef image1d_buffer_t img1dbuff_t;

// image2d_t
typedef image2d_t img2d_t;

// image2d_array_t
typedef image2d_array_t img2darr_t;

// image3d_t
typedef image3d_t img3d_t;

// sampler_t
typedef sampler_t smp_t;

// event_t
typedef event_t evt_t;

#if __OPENCL_VERSION__ >= 200

// clk_event_t
typedef clk_event_t clkevt_t;

// queue_t
typedef queue_t q_t;

// ndrange_t
typedef ndrange_t range_t;

// reserve_id_t
typedef reserve_id_t reserveid_t;

// image2d_depth_t
typedef image2d_depth_t img2ddep_t;

// image2d_array_depth_t
typedef image2d_array_depth_t img2darr_dep_t;

// image2d_msaa_t
typedef image2d_msaa_t img2dmsaa_t;

// image2d_array_msaa_t
typedef image2d_array_msaa_t img2darrmsaa_t;

// image2d_msaa_depth_t
typedef image2d_msaa_depth_t img2dmsaadep_t;

// image2d_array_msaa_depth_t
typedef image2d_array_msaa_depth_t img2darrmsaadep_t;

#endif
