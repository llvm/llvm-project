# OpenCL built-in library: type conversion functions
#
# Copyright (c) 2013 Victor Oliveira <victormatheus@gmail.com>
# Copyright (c) 2013 Jesse Towner <jessetowner@lavabit.com>
# Copyright (c) 2024 Romaric Jodin <rjodin@chromium.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# This script generates the file convert_type.cl, which contains all of the
# OpenCL functions in the form:
#
# convert_<destTypen><_sat><_roundingMode>(<sourceTypen>)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--clc", action="store_true", help="Generate clc internal conversions"
)
parser.add_argument(
    "--clspv", action="store_true", help="Generate the clspv variant of the code"
)
args = parser.parse_args()

clc = args.clc
clspv = args.clspv

types = [
    "char",
    "uchar",
    "short",
    "ushort",
    "int",
    "uint",
    "long",
    "ulong",
    "half",
    "float",
    "double",
]
int_types = ["char", "uchar", "short", "ushort", "int", "uint", "long", "ulong"]
unsigned_types = ["uchar", "ushort", "uint", "ulong"]
float_types = ["half", "float", "double"]
int64_types = ["long", "ulong"]
float64_types = ["double"]
float16_types = ["half"]
vector_sizes = ["", "2", "3", "4", "8", "16"]
half_sizes = [("2", ""), ("4", "2"), ("8", "4"), ("16", "8")]

saturation = ["", "_sat"]
rounding_modes = ["_rtz", "_rte", "_rtp", "_rtn"]

bool_type = {
    "char": "char",
    "uchar": "char",
    "short": "short",
    "ushort": "short",
    "int": "int",
    "uint": "int",
    "long": "long",
    "ulong": "long",
    "half": "short",
    "float": "int",
    "double": "long",
}

unsigned_type = {
    "char": "uchar",
    "uchar": "uchar",
    "short": "ushort",
    "ushort": "ushort",
    "int": "uint",
    "uint": "uint",
    "long": "ulong",
    "ulong": "ulong",
}

sizeof_type = {
    "char": 1,
    "uchar": 1,
    "short": 2,
    "ushort": 2,
    "int": 4,
    "uint": 4,
    "long": 8,
    "ulong": 8,
    "half": 2,
    "float": 4,
    "double": 8,
}

limit_max = {
    "char": "CHAR_MAX",
    "uchar": "UCHAR_MAX",
    "short": "SHRT_MAX",
    "ushort": "USHRT_MAX",
    "int": "INT_MAX",
    "uint": "UINT_MAX",
    "long": "LONG_MAX",
    "ulong": "ULONG_MAX",
    "half": "0x1.ffcp+15",
}

limit_min = {
    "char": "CHAR_MIN",
    "uchar": "0",
    "short": "SHRT_MIN",
    "ushort": "0",
    "int": "INT_MIN",
    "uint": "0",
    "long": "LONG_MIN",
    "ulong": "0",
    "half": "-0x1.ffcp+15",
}


def conditional_guard(src, dst):
    int64_count = 0
    float64_count = 0
    float16_count = 0
    if src in int64_types:
        int64_count = int64_count + 1
    elif src in float64_types:
        float64_count = float64_count + 1
    elif src in float16_types:
        float16_count = float16_count + 1
    if dst in int64_types:
        int64_count = int64_count + 1
    elif dst in float64_types:
        float64_count = float64_count + 1
    elif dst in float16_types:
        float16_count = float16_count + 1
    if float64_count > 0 and float16_count > 0:
        print("#if defined(cl_khr_fp16) && defined(cl_khr_fp64)")
        return True
    elif float64_count > 0:
        # In embedded profile, if cl_khr_fp64 is supported cles_khr_int64 has to be
        print("#ifdef cl_khr_fp64")
        return True
    elif float16_count > 0:
        print("#if defined cl_khr_fp16")
        return True
    elif int64_count > 0:
        print("#if defined cles_khr_int64 || !defined(__EMBEDDED_PROFILE__)")
        return True
    return False


nl = "\n"
includes = []
if not clc:
    includes = ["<clc/clc.h>"]
else:
    includes = sorted([
        "<clc/internal/clc.h>",
        "<clc/integer/definitions.h>",
        "<clc/float/definitions.h>",
        "<clc/integer/clc_abs.h>",
        "<clc/common/clc_sign.h>",
        "<clc/shared/clc_clamp.h>",
        "<clc/shared/clc_min.h>",
        "<clc/shared/clc_max.h>",
        "<clc/math/clc_fabs.h>",
        "<clc/math/clc_rint.h>",
        "<clc/math/clc_ceil.h>",
        "<clc/math/clc_floor.h>",
        "<clc/math/clc_nextafter.h>",
        "<clc/relational/clc_select.h>",
    ])

print(
    f"""/* !!!! AUTOGENERATED FILE generated by convert_type.py !!!!!

   DON'T CHANGE THIS FILE. MAKE YOUR CHANGES TO convert_type.py AND RUN:
   $ ./generate-conversion-type-cl.sh

   OpenCL type conversion functions

   Copyright (c) 2013 Victor Oliveira <victormatheus@gmail.com>
   Copyright (c) 2013 Jesse Towner <jessetowner@lavabit.com>

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

{nl.join(['#include ' + f for f in includes])}
#include <clc/clc_convert.h>

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#if defined(__EMBEDDED_PROFILE__) && !defined(cles_khr_int64)
#error Embedded profile that supports cl_khr_fp64 also has to support cles_khr_int64
#endif

#endif

#ifdef cles_khr_int64
#pragma OPENCL EXTENSION cles_khr_int64 : enable
#endif

"""
)


#
# Default Conversions
#
# All conversions are in accordance with the OpenCL specification,
# which cites the C99 conversion rules.
#
# Casting from floating point to integer results in conversions
# with truncation, so it should be suitable for the default convert
# functions.
#
# Conversions from integer to floating-point, and floating-point to
# floating-point through casting is done with the default rounding
# mode. While C99 allows dynamically changing the rounding mode
# during runtime, it is not a supported feature in OpenCL according
# to Section 7.1 - Rounding Modes in the OpenCL 1.2 specification.
#
# Therefore, we can assume for optimization purposes that the
# rounding mode is fixed to round-to-nearest-even. Platform target
# authors should ensure that the rounding-control registers remain
# in this state, and that this invariant holds.
#
# Also note, even though the OpenCL specification isn't entirely
# clear on this matter, we implement all rounding mode combinations
# even for integer-to-integer conversions. When such a conversion
# is used, the rounding mode is ignored.
#
def print_passthru_conversion(src_ty, dst_ty, fn_name):
    print(
        f"""_CLC_DEF _CLC_OVERLOAD {dst_ty} {fn_name}({src_ty} x) {{
  return __clc_{fn_name}(x);
}}
"""
    )


def generate_default_conversion(src, dst, mode):
    close_conditional = conditional_guard(src, dst)

    for size in vector_sizes:
        if not size:
            if clc:
                print(
                    f"""_CLC_DEF _CLC_OVERLOAD {dst} __clc_convert_{dst}{mode}({src} x) {{
  return ({dst})x;
}}
"""
                )
            else:
                print_passthru_conversion(src, dst, f"convert_{dst}{mode}")
        else:
            if clc:
                print(
                    f"""_CLC_DEF _CLC_OVERLOAD {dst}{size} __clc_convert_{dst}{size}{mode}({src}{size} x) {{
  return __builtin_convertvector(x, {dst}{size});
}}
"""
                )
            else:
                print_passthru_conversion(
                    f"{src}{size}", f"{dst}{size}", f"convert_{dst}{size}{mode}"
                )

    if close_conditional:
        print("#endif")


# Do not generate user-facing default conversions for clspv as they are handled
# natively
if clc or not clspv:
    for src in types:
        for dst in types:
            generate_default_conversion(src, dst, "")

for src in int_types:
    for dst in int_types:
        for mode in rounding_modes:
            # Do not generate user-facing "_rte" conversions for clspv as they
            # are handled natively
            if clspv and not clc and mode == "_rte":
                continue
            generate_default_conversion(src, dst, mode)

#
# Saturated Conversions To Integers


# These functions are dependent on the unsaturated conversion functions
# generated above, and use clamp, max, min, and select to eliminate
# branching and vectorize the conversions.
#
# Again, as above, we allow all rounding modes for integer-to-integer
# conversions with saturation.
#
def generate_saturated_conversion(src, dst, size):
    # Header
    close_conditional = conditional_guard(src, dst)

    dstn = f"{dst}{size}"
    srcn = f"{src}{size}"

    if not clc:
        print_passthru_conversion(f"{srcn}", f"{dstn}", f"convert_{dstn}_sat")
        if close_conditional:
            print("#endif")
        return

    print(f"_CLC_DEF _CLC_OVERLOAD {dstn} __clc_convert_{dstn}_sat({srcn} x) {{")

    # FIXME: This is a work around for lack of select function with signed
    # third argument when the first two arguments are unsigned types. We cast
    # to the signed type for sign-extension, then do a bitcast to the unsigned
    # type.
    if dst in unsigned_types:
        bool_prefix = f"__clc_as_{dstn}(__clc_convert_{bool_type[dst]}{size}"
        bool_suffix = ")"
    else:
        bool_prefix = f"__clc_convert_{bool_type[dst]}{size}"
        bool_suffix = ""

    dst_max = limit_max[dst]
    dst_min = limit_min[dst]

    # Body
    if src == dst:
        # Conversion between same types
        print("  return x;")

    elif src in float_types:

        if clspv:
            # Conversion from float to int
            print(
                f"""  {dstn} y = __clc_convert_{dstn}(x);
  y = __clc_select(y, ({dstn}){dst_min}, {bool_prefix}(x <= ({srcn}){dst_min}){bool_suffix});
  y = __clc_select(y, ({dstn}){dst_max}, {bool_prefix}(x >= ({srcn}){dst_max}){bool_suffix});
  return y;"""
            )
        else:
            # Conversion from float to int
            print(
                f"""  {dstn} y = __clc_convert_{dstn}(x);
  y = __clc_select(y, ({dstn}){dst_min}, {bool_prefix}(x < ({srcn}){dst_min}){bool_suffix});
  y = __clc_select(y, ({dstn}){dst_max}, {bool_prefix}(x > ({srcn}){dst_max}){bool_suffix});
  return y;"""
            )
    else:

        # Integer to integer convesion with sizeof(src) == sizeof(dst)
        if sizeof_type[src] == sizeof_type[dst]:
            if src in unsigned_types:
                print(f"  x = __clc_min(x, ({src}){dst_max});")
            else:
                print(f"  x = __clc_max(x, ({src})0);")

        # Integer to integer conversion where sizeof(src) > sizeof(dst)
        elif sizeof_type[src] > sizeof_type[dst]:
            if src in unsigned_types:
                print(f"  x = __clc_min(x, ({src}){dst_max});")
            else:
                print(f"  x = __clc_clamp(x, ({src}){dst_min}, ({src}){dst_max});")

        # Integer to integer conversion where sizeof(src) < sizeof(dst)
        elif src not in unsigned_types and dst in unsigned_types:
            print(f"  x = __clc_max(x, ({src})0);")

        print(f"  return __clc_convert_{dstn}(x);")

    # Footer
    print("}")
    if close_conditional:
        print("#endif")


for src in types:
    for dst in int_types:
        for size in vector_sizes:
            generate_saturated_conversion(src, dst, size)


def generate_saturated_conversion_with_rounding(src, dst, size, mode):
    # Header
    close_conditional = conditional_guard(src, dst)

    dstn = f"{dst}{size}"
    srcn = f"{src}{size}"

    if not clc:
        print_passthru_conversion(f"{srcn}", f"{dstn}", f"convert_{dstn}_sat{mode}")
    else:
        # Body
        print(
            f"""_CLC_DEF _CLC_OVERLOAD {dstn} __clc_convert_{dstn}_sat{mode}({srcn} x) {{
  return __clc_convert_{dstn}_sat(x);
}}
"""
        )

    # Footer
    if close_conditional:
        print("#endif")


for src in int_types:
    for dst in int_types:
        for size in vector_sizes:
            for mode in rounding_modes:
                generate_saturated_conversion_with_rounding(src, dst, size, mode)


#
# Conversions To/From Floating-Point With Rounding
#
# Note that we assume as above that casts from floating-point to
# integer are done with truncation, and that the default rounding
# mode is fixed to round-to-nearest-even, as per C99 and OpenCL
# rounding rules.
#
# These functions rely on the use of abs, ceil, fabs, floor,
# nextafter, sign, rint and the above generated conversion functions.
#
# Only conversions to integers can have saturation.
#
def generate_float_conversion(src, dst, size, mode, sat):
    # Header
    close_conditional = conditional_guard(src, dst)

    dstn = f"{dst}{size}"
    srcn = f"{src}{size}"
    booln = f"{bool_type[dst]}{size}"
    src_max = limit_max[src] if src in limit_max else ""
    dst_min = limit_min[dst] if dst in limit_min else ""

    if not clc:
        print_passthru_conversion(f"{srcn}", f"{dstn}", f"convert_{dstn}{sat}{mode}")
        # Footer
        if close_conditional:
            print("#endif")
        return

    print(f"_CLC_DEF _CLC_OVERLOAD {dstn} __clc_convert_{dstn}{sat}{mode}({srcn} x) {{")

    # Perform conversion
    if dst in int_types:
        if mode == "_rte":
            print("  x = __clc_rint(x);")
        elif mode == "_rtp":
            print("  x = __clc_ceil(x);")
        elif mode == "_rtn":
            print("  x = __clc_floor(x);")
        print(f"  return __clc_convert_{dstn}{sat}(x);")
    elif mode == "_rte":
        print(f"  return __clc_convert_{dstn}(x);")
    else:
        print(f"  {dstn} r = __clc_convert_{dstn}(x);")
        if clspv:
            print(f"  {srcn} y = __clc_convert_{srcn}_sat(r);")
        else:
            print(f"  {srcn} y = __clc_convert_{srcn}(r);")
        if mode == "_rtz":
            if src in int_types:
                usrcn = f"{unsigned_type[src]}{size}"
                print(f"  {usrcn} abs_x = __clc_abs(x);")
                print(f"  {usrcn} abs_y = __clc_abs(y);")
            else:
                print(f"  {srcn} abs_x = __clc_fabs(x);")
                print(f"  {srcn} abs_y = __clc_fabs(y);")
            print(f"  {booln} c = __clc_convert_{booln}(abs_y > abs_x);")
            if clspv and sizeof_type[src] >= 4 and src in int_types:
                print(f"  c = c || __clc_convert_{booln}(({srcn}){src_max} == x);")
            print(
                f"  {dstn} sel = __clc_select(r, __clc_nextafter(r, __clc_sign(r) * ({dstn})-INFINITY), c);"
            )
            if dst == "half" and src in int_types and sizeof_type[src] >= 2:
                dst_max = limit_max[dst]
                # short is 16 bits signed, so the maximum value rounded to zero
                # is 0x1.ffcp+14 (0x1p+15 == 32768 > 0x7fff == 32767)
                if src == "short":
                    dst_max = "0x1.ffcp+14"
                print(
                    f"  return __clc_clamp(sel, ({dstn}){dst_min}, ({dstn}){dst_max});"
                )
            else:
                print("  return sel;")
        if mode == "_rtp":
            print(
                f"  {dstn} sel = __clc_select(r, __clc_nextafter(r, ({dstn})INFINITY), __clc_convert_{booln}(y < x));"
            )
            if dst == "half" and src in int_types and sizeof_type[src] >= 2:
                print(f"  return __clc_max(sel, ({dstn}){dst_min});")
            else:
                print("  return sel;")
        if mode == "_rtn":
            print(f"  {booln} c = __clc_convert_{booln}(y > x);")
            if clspv and sizeof_type[src] >= 4 and src in int_types:
                print(f"  c = c || __clc_convert_{booln}(({srcn}){src_max} == x);")
            print(
                f"  {dstn} sel = __clc_select(r, __clc_nextafter(r, ({dstn})-INFINITY), c);"
            )
            if dst == "half" and src in int_types and sizeof_type[src] >= 2:
                dst_max = limit_max[dst]
                # short is 16 bits signed, so the maximum value rounded to
                # negative infinity is 0x1.ffcp+14 (0x1p+15 == 32768 > 0x7fff
                # == 32767)
                if src == "short":
                    dst_max = "0x1.ffcp+14"
                print(f"  return __clc_min(sel, ({dstn}){dst_max});")
            else:
                print("  return sel;")

    # Footer
    print("}")
    if close_conditional:
        print("#endif")


for src in float_types:
    for dst in int_types:
        for size in vector_sizes:
            for mode in rounding_modes:
                for sat in saturation:
                    generate_float_conversion(src, dst, size, mode, sat)


for src in types:
    for dst in float_types:
        for size in vector_sizes:
            for mode in rounding_modes:
                # Do not generate user-facing "_rte" conversions for clspv as
                # they are handled natively
                if clspv and not clc and mode == "_rte":
                    continue
                generate_float_conversion(src, dst, size, mode, "")
