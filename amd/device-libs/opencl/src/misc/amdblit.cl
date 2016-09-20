/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#if !defined NO_BLIT

static const uint SplitCount = 3;

__attribute__((always_inline)) void
__amd_copyBufferToImage(
    __global uint *src,
    __write_only image2d_array_t dst,
    ulong4 srcOrigin,
    int4 dstOrigin,
    int4 size,
    uint4 format,
    ulong4 pitch)
{
    ulong idxSrc;
    int4 coordsDst;
    uint4 pixel;
    __global uint* srcUInt = src;
    __global ushort* srcUShort = (__global ushort*)src;
    __global uchar* srcUChar  = (__global uchar*)src;
    ushort tmpUShort;
    uint tmpUInt;

    coordsDst.x = get_global_id(0);
    coordsDst.y = get_global_id(1);
    coordsDst.z = get_global_id(2);
    coordsDst.w = 0;

    if ((coordsDst.x >= size.x) ||
        (coordsDst.y >= size.y) ||
        (coordsDst.z >= size.z)) {
        return;
    }

    idxSrc = (coordsDst.z * pitch.y +
       coordsDst.y * pitch.x + coordsDst.x) *
       format.z + srcOrigin.x;

    coordsDst.x += dstOrigin.x;
    coordsDst.y += dstOrigin.y;
    coordsDst.z += dstOrigin.z;

    // Check components
    switch (format.x) {
    case 1:
        // Check size
        if (format.y == 1) {
            pixel.x = (uint)srcUChar[idxSrc];
        }
        else if (format.y == 2) {
            pixel.x = (uint)srcUShort[idxSrc];
        }
        else {
            pixel.x = srcUInt[idxSrc];
        }
    break;
    case 2:
        // Check size
        if (format.y == 1) {
            tmpUShort = srcUShort[idxSrc];
            pixel.x = (uint)(tmpUShort & 0xff);
            pixel.y = (uint)(tmpUShort >> 8);
        }
        else if (format.y == 2) {
            tmpUInt = srcUInt[idxSrc];
            pixel.x = (tmpUInt & 0xffff);
            pixel.y = (tmpUInt >> 16);
        }
        else {
            pixel.x = srcUInt[idxSrc++];
            pixel.y = srcUInt[idxSrc];
        }
    break;
    case 4:
        // Check size
        if (format.y == 1) {
            tmpUInt = srcUInt[idxSrc];
            pixel.x = tmpUInt & 0xff;
            pixel.y = (tmpUInt >> 8) & 0xff;
            pixel.z = (tmpUInt >> 16) & 0xff;
            pixel.w = (tmpUInt >> 24) & 0xff;
        }
        else if (format.y == 2) {
            tmpUInt = srcUInt[idxSrc++];
            pixel.x = tmpUInt & 0xffff;
            pixel.y = (tmpUInt >> 16);
            tmpUInt = srcUInt[idxSrc];
            pixel.z = tmpUInt & 0xffff;
            pixel.w = (tmpUInt >> 16);
        }
        else {
            pixel.x = srcUInt[idxSrc++];
            pixel.y = srcUInt[idxSrc++];
            pixel.z = srcUInt[idxSrc++];
            pixel.w = srcUInt[idxSrc];
        }
    break;
    }
    // Write the final pixel
    write_imageui(dst, coordsDst, pixel);
}

__attribute__((always_inline)) void
__amd_copyImageToBuffer(
    __read_only image2d_array_t src,
    __global uint* dstUInt,
    __global ushort* dstUShort,
    __global uchar* dstUChar,
    int4 srcOrigin,
    ulong4 dstOrigin,
    int4 size,
    uint4 format,
    ulong4 pitch)
{
    ulong idxDst;
    int4 coordsSrc;
    uint4 texel;

    coordsSrc.x = get_global_id(0);
    coordsSrc.y = get_global_id(1);
    coordsSrc.z = get_global_id(2);
    coordsSrc.w = 0;

    if ((coordsSrc.x >= size.x) ||
        (coordsSrc.y >= size.y) ||
        (coordsSrc.z >= size.z)) {
        return;
    }

    idxDst = (coordsSrc.z * pitch.y + coordsSrc.y * pitch.x +
        coordsSrc.x) * format.z + dstOrigin.x;

    coordsSrc.x += srcOrigin.x;
    coordsSrc.y += srcOrigin.y;
    coordsSrc.z += srcOrigin.z;

    texel = read_imageui(src, coordsSrc);

    // Check components
    switch (format.x) {
    case 1:
        // Check size
        switch (format.y) {
        case 1:
            dstUChar[idxDst] = (uchar)texel.x;
            break;
        case 2:
            dstUShort[idxDst] = (ushort)texel.x;
            break;
        case 4:
            dstUInt[idxDst] = texel.x;
            break;
        }
    break;
    case 2:
        // Check size
        switch (format.y) {
        case 1:
            dstUShort[idxDst] = (ushort)texel.x |
               ((ushort)texel.y << 8);
            break;
        case 2:
            dstUInt[idxDst] = texel.x | (texel.y << 16);
            break;
        case 4:
            dstUInt[idxDst++] = texel.x;
            dstUInt[idxDst] = texel.y;
            break;
        }
    break;
    case 4:
        // Check size
        switch (format.y) {
        case 1:
            dstUInt[idxDst] = (uint)texel.x |
               (texel.y << 8) |
               (texel.z << 16) |
               (texel.w << 24);
            break;
        case 2:
            dstUInt[idxDst++] = texel.x | (texel.y << 16);
            dstUInt[idxDst] = texel.z | (texel.w << 16);
            break;
        case 4:
            dstUInt[idxDst++] = texel.x;
            dstUInt[idxDst++] = texel.y;
            dstUInt[idxDst++] = texel.z;
            dstUInt[idxDst] = texel.w;
            break;
        }
    break;
    }
}

__attribute__((always_inline)) void
__amd_copyImage(
    __read_only image2d_array_t src,
    __write_only image2d_array_t dst,
    int4 srcOrigin,
    int4 dstOrigin,
    int4 size)
{
    int4    coordsDst;
    int4    coordsSrc;

    coordsDst.x = get_global_id(0);
    coordsDst.y = get_global_id(1);
    coordsDst.z = get_global_id(2);
    coordsDst.w = 0;

    if ((coordsDst.x >= size.x) ||
        (coordsDst.y >= size.y) ||
        (coordsDst.z >= size.z)) {
        return;
    }

    coordsSrc = srcOrigin + coordsDst;
    coordsDst += dstOrigin;

    uint4  texel;
    texel = read_imageui(src, coordsSrc);
    write_imageui(dst, coordsDst, texel);
}

__attribute__((always_inline)) void
__amd_copyImage1DA(
    __read_only image2d_array_t src,
    __write_only image2d_array_t dst,
    int4 srcOrigin,
    int4 dstOrigin,
    int4 size)
{
    int4 coordsDst;
    int4 coordsSrc;

    coordsDst.x = get_global_id(0);
    coordsDst.y = get_global_id(1);
    coordsDst.z = get_global_id(2);
    coordsDst.w = 0;

    if ((coordsDst.x >= size.x) ||
        (coordsDst.y >= size.y) ||
        (coordsDst.z >= size.z)) {
        return;
    }

    coordsSrc = srcOrigin + coordsDst;
    coordsDst += dstOrigin;
    if (srcOrigin.w != 0) {
       coordsSrc.z = coordsSrc.y;
       coordsSrc.y = 0;
    }
    if (dstOrigin.w != 0) {
       coordsDst.z = coordsDst.y;
       coordsDst.y = 0;
    }

    uint4  texel;
    texel = read_imageui(src, coordsSrc);
    write_imageui(dst, coordsDst, texel);
}

__attribute__((always_inline)) void
__amd_copyBufferRect(
    __global uchar* src,
    __global uchar* dst,
    ulong4 srcRect,
    ulong4 dstRect,
    ulong4 size)
{
    ulong x = get_global_id(0);
    ulong y = get_global_id(1);
    ulong z = get_global_id(2);

    if ((x >= size.x) ||
        (y >= size.y) ||
        (z >= size.z)) {
        return;
    }

    ulong offsSrc = srcRect.z + x + y * srcRect.x + z * srcRect.y;
    ulong offsDst = dstRect.z + x + y * dstRect.x + z * dstRect.y;

    dst[offsDst] = src[offsSrc];
}

__attribute__((always_inline)) void
__amd_copyBufferRectAligned(
    __global uint* src,
    __global uint* dst,
    ulong4 srcRect,
    ulong4 dstRect,
    ulong4 size)
{
    ulong x = get_global_id(0);
    ulong y = get_global_id(1);
    ulong z = get_global_id(2);

    if ((x >= size.x) ||
        (y >= size.y) ||
        (z >= size.z)) {
        return;
    }

    ulong offsSrc = srcRect.z + x + y * srcRect.x + z * srcRect.y;
    ulong offsDst = dstRect.z + x + y * dstRect.x + z * dstRect.y;

    if (size.w == 16) {
        __global uint4* src4 = (__global uint4*)src;
        __global uint4* dst4 = (__global uint4*)dst;
        dst4[offsDst] = src4[offsSrc];
    }
    else {
        dst[offsDst] = src[offsSrc];
    }
}

__attribute__((always_inline)) void
__amd_copyBuffer(
    __global uchar* srcI,
    __global uchar* dstI,
    ulong srcOrigin,
    ulong dstOrigin,
    ulong size,
    uint remain)
{
    ulong id = get_global_id(0);

    if (id >= size) {
        return;
    }

    __global uchar* src = srcI + srcOrigin;
    __global uchar* dst = dstI + dstOrigin;

    if (remain == 8) {
        dst[id] = src[id];
    }
    else {
        if (id < (size - 1)) {
            __global uint* srcD = (__global uint*)(src);
            __global uint* dstD = (__global uint*)(dst);
            dstD[id] = srcD[id];
        }
        else {
            for (uint i = 0; i < remain; ++i) {
                dst[id * 4 + i] = src[id * 4 + i];
            }
        }
    }
}

__attribute__((always_inline)) void
__amd_copyBufferAligned(
    __global uint* src,
    __global uint* dst,
    ulong srcOrigin,
    ulong dstOrigin,
    ulong size,
    uint alignment)
{
    ulong id = get_global_id(0);

    if (id >= size) {
        return;
    }

    ulong   offsSrc = id + srcOrigin;
    ulong   offsDst = id + dstOrigin;

    if (alignment == 16) {
        __global uint4* src4 = (__global uint4*)src;
        __global uint4* dst4 = (__global uint4*)dst;
        dst4[offsDst] = src4[offsSrc];
    }
    else {
        dst[offsDst] = src[offsSrc];
    }
}

__attribute__((always_inline)) void
__amd_fillBuffer(
    __global uchar* bufUChar,
    __global uint* bufUInt,
    __constant uchar* pattern,
    uint patternSize,
    ulong offset,
    ulong size)
{
    ulong id = get_global_id(0);

    if (id >= size) {
        return;
    }

    if (bufUInt) {
       __global uint* element = &bufUInt[offset + id * patternSize];
       __constant uint*  pt = (__constant uint*)pattern;

        for (uint i = 0; i < patternSize; ++i) {
            element[i] = pt[i];
        }
    }
    else {
        __global uchar* element = &bufUChar[offset + id * patternSize];

        for (uint i = 0; i < patternSize; ++i) {
            element[i] = pattern[i];
        }
    }
}

__attribute__((always_inline)) void
__amd_fillImage(
    __write_only image2d_array_t image,
    float4 patternFLOAT4,
    int4 patternINT4,
    uint4 patternUINT4,
    int4 origin,
    int4 size,
    uint type)
{
    int4  coords;

    coords.x = get_global_id(0);
    coords.y = get_global_id(1);
    coords.z = get_global_id(2);
    coords.w = 0;

    if ((coords.x >= size.x) ||
        (coords.y >= size.y) ||
        (coords.z >= size.z)) {
        return;
    }

    coords += origin;

    int SizeX = get_global_size(0);
    int AdjustedSizeX = size.x + origin.x;

    for (uint i = 0; i < SplitCount; ++i) {
        // Check components
        switch (type) {
        case 0:
            write_imagef(image, coords, patternFLOAT4);
            break;
        case 1:
            write_imagei(image, coords, patternINT4);
            break;
        case 2:
            write_imageui(image, coords, patternUINT4);
            break;
        }
        coords.x += SizeX;
        if (coords.x >= AdjustedSizeX) return;
    }
}

#endif

