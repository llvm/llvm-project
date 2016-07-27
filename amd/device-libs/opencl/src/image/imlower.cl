/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

#define CATTR __attribute__((always_inline, const))

struct __opencl_sampler_t;

struct __opencl_image1d_ro_t;
struct __opencl_image1d_array_ro_t;
struct __opencl_image1d_buffer_ro_t;
struct __opencl_image2d_ro_t;
struct __opencl_image2d_array_ro_t;
struct __opencl_image2d_depth_ro_t;
struct __opencl_image2d_array_depth_ro_t;
struct __opencl_image3d_ro_t;

struct __opencl_image1d_wo_t;
struct __opencl_image1d_array_wo_t;
struct __opencl_image1d_buffer_wo_t;
struct __opencl_image2d_wo_t;
struct __opencl_image2d_array_wo_t;
struct __opencl_image2d_depth_wo_t;
struct __opencl_image2d_array_depth_wo_t;
struct __opencl_image3d_wo_t;

struct __opencl_image1d_rw_t;
struct __opencl_image1d_array_rw_t;
struct __opencl_image1d_buffer_rw_t;
struct __opencl_image2d_rw_t;
struct __opencl_image2d_array_rw_t;
struct __opencl_image2d_depth_rw_t;
struct __opencl_image2d_array_depth_rw_t;
struct __opencl_image3d_rw_t;

CATTR SSHARP __lower_sampler(__constant struct __opencl_sampler_t * s) { return (SSHARP)s; }

CATTR TSHARP __lower_ro_1D(__constant struct __opencl_image1d_ro_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_ro_1Da(__constant struct __opencl_image1d_array_ro_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_ro_1Db(__constant struct __opencl_image1d_buffer_ro_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_ro_2D(__constant struct __opencl_image2d_ro_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_ro_2Da(__constant struct __opencl_image2d_array_ro_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_ro_2Dd(__constant struct __opencl_image2d_depth_ro_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_ro_2Dad(__constant struct __opencl_image2d_array_depth_ro_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_ro_3D(__constant struct __opencl_image3d_ro_t * i) { return (TSHARP)i; }

CATTR TSHARP __lower_wo_1D(__constant struct __opencl_image1d_wo_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_wo_1Da(__constant struct __opencl_image1d_array_wo_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_wo_1Db(__constant struct __opencl_image1d_buffer_wo_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_wo_2D(__constant struct __opencl_image2d_wo_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_wo_2Da(__constant struct __opencl_image2d_array_wo_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_wo_2Dd(__constant struct __opencl_image2d_depth_wo_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_wo_2Dad(__constant struct __opencl_image2d_array_depth_wo_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_wo_3D(__constant struct __opencl_image3d_wo_t * i) { return (TSHARP)i; }

CATTR TSHARP __lower_rw_1D(__constant struct __opencl_image1d_rw_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_rw_1Da(__constant struct __opencl_image1d_array_rw_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_rw_1Db(__constant struct __opencl_image1d_buffer_rw_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_rw_2D(__constant struct __opencl_image2d_rw_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_rw_2Da(__constant struct __opencl_image2d_array_rw_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_rw_2Dd(__constant struct __opencl_image2d_depth_rw_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_rw_2Dad(__constant struct __opencl_image2d_array_depth_rw_t * i) { return (TSHARP)i; }
CATTR TSHARP __lower_rw_3D(__constant struct __opencl_image3d_rw_t * i) { return (TSHARP)i; }

