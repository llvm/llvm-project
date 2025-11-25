

#include "devenq.h"

#define ATTR __attribute__((overloadable, always_inline, const))

// 1D variants

ATTR ndrange_t
ndrange_1D(size_t gws)
{
    ndrange_t ret;
    ret.workDimension = 1;
    ret.globalWorkOffset[0] = 0;
    ret.globalWorkOffset[1] = 0;
    ret.globalWorkOffset[2] = 0;
    ret.globalWorkSize[0] = gws;
    ret.globalWorkSize[1] = 1;
    ret.globalWorkSize[2] = 1;
    ret.localWorkSize[0] = min(gws, (size_t)64);
    ret.localWorkSize[1] = 1;
    ret.localWorkSize[2] = 1;
    return ret;
}

ATTR ndrange_t
ndrange_1D(size_t gws, size_t lws)
{
    ndrange_t ret;
    ret.workDimension = 1;
    ret.globalWorkOffset[0] = 0;
    ret.globalWorkOffset[1] = 0;
    ret.globalWorkOffset[2] = 0;
    ret.globalWorkSize[0] = gws;
    ret.globalWorkSize[1] = 1;
    ret.globalWorkSize[2] = 1;
    ret.localWorkSize[0] = lws;
    ret.localWorkSize[1] = 1;
    ret.localWorkSize[2] = 1;
    return ret;
}

ATTR ndrange_t
ndrange_1D(size_t goff, size_t gws, size_t lws)
{
    ndrange_t ret;
    ret.workDimension = 1;
    ret.globalWorkOffset[0] = goff;
    ret.globalWorkOffset[1] = 0;
    ret.globalWorkOffset[2] = 0;
    ret.globalWorkSize[0] = gws;
    ret.globalWorkSize[1] = 1;
    ret.globalWorkSize[2] = 1;
    ret.localWorkSize[0] = lws;
    ret.localWorkSize[1] = 1;
    ret.localWorkSize[2] = 1;
    return ret;
}

// 2D variants

ATTR ndrange_t
ndrange_2D(const size_t gws[2])
{
    ndrange_t ret;
    ret.workDimension = 2;
    ret.globalWorkOffset[0] = 0;
    ret.globalWorkOffset[1] = 0;
    ret.globalWorkOffset[2] = 0;
    ret.globalWorkSize[0] = gws[0];
    ret.globalWorkSize[1] = gws[1];
    ret.globalWorkSize[2] = 1;
    ret.localWorkSize[0] = min(gws[0], (size_t)8);
    ret.localWorkSize[1] = min(gws[1], (size_t)8);
    ret.localWorkSize[2] = 1;
    return ret;
}

ATTR ndrange_t
ndrange_2D(const size_t gws[2], const size_t lws[2])
{
    ndrange_t ret;
    ret.workDimension = 2;
    ret.globalWorkOffset[0] = 0;
    ret.globalWorkOffset[1] = 0;
    ret.globalWorkOffset[2] = 0;
    ret.globalWorkSize[0] = gws[0];
    ret.globalWorkSize[1] = gws[1];
    ret.globalWorkSize[2] = 1;
    ret.localWorkSize[0] = lws[0];
    ret.localWorkSize[1] = lws[1];
    ret.localWorkSize[2] = 1;
    return ret;
}

ATTR ndrange_t
ndrange_2D(const size_t goff[2], const size_t gws[2], const size_t lws[2])
{
    ndrange_t ret;
    ret.workDimension = 2;
    ret.globalWorkOffset[0] = goff[0];
    ret.globalWorkOffset[1] = goff[1];
    ret.globalWorkOffset[2] = 0;
    ret.globalWorkSize[0] = gws[0];
    ret.globalWorkSize[1] = gws[1];
    ret.globalWorkSize[2] = 1;
    ret.localWorkSize[0] = lws[0];
    ret.localWorkSize[1] = lws[1];
    ret.localWorkSize[2] = 1;
    return ret;
}

// 3D variants

ATTR ndrange_t
ndrange_3D(const size_t gws[3])
{
    ndrange_t ret;
    ret.workDimension = 3;
    ret.globalWorkOffset[0] = 0;
    ret.globalWorkOffset[1] = 0;
    ret.globalWorkOffset[2] = 0;
    ret.globalWorkSize[0] = gws[0];
    ret.globalWorkSize[1] = gws[1];
    ret.globalWorkSize[2] = gws[2];
    ret.localWorkSize[0] = min(gws[0], (size_t)4);
    ret.localWorkSize[1] = min(gws[1], (size_t)4);
    ret.localWorkSize[2] = min(gws[2], (size_t)4);
    return ret;
}

ATTR ndrange_t
ndrange_3D(const size_t gws[3], const size_t lws[3])
{
    ndrange_t ret;
    ret.workDimension = 3;
    ret.globalWorkOffset[0] = 0;
    ret.globalWorkOffset[1] = 0;
    ret.globalWorkOffset[2] = 0;
    ret.globalWorkSize[0] = gws[0];
    ret.globalWorkSize[1] = gws[1];
    ret.globalWorkSize[2] = gws[2];
    ret.localWorkSize[0] = lws[0];
    ret.localWorkSize[1] = lws[1];
    ret.localWorkSize[2] = lws[2];
    return ret;
}

ATTR ndrange_t
ndrange_3D(const size_t goff[3], const size_t gws[3], const size_t lws[3])
{
    ndrange_t ret;
    ret.workDimension = 3;
    ret.globalWorkOffset[0] = goff[0];
    ret.globalWorkOffset[1] = goff[1];
    ret.globalWorkOffset[2] = goff[2];
    ret.globalWorkSize[0] = gws[0];
    ret.globalWorkSize[1] = gws[1];
    ret.globalWorkSize[2] = gws[2];
    ret.localWorkSize[0] = lws[0];
    ret.localWorkSize[1] = lws[1];
    ret.localWorkSize[2] = lws[2];
    return ret;
}

