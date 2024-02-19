"""
//===-- Table Generator for Ryu Printf ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


This file is used to generate the tables of values in 
src/__support/ryu_constants.h and ryu_long_double constants.h. To use it, set
the constants at the top of the file to the values you want to use for the Ryu
algorithm, then run this file. It will output the appropriate tables to stdout,
so it's recommended to pipe stdout to a file. The following is a brief
explenation of each of the constants.

BLOCK_SIZE:
    Default: 9
    The number of digits that will be calculated together in a block.
    Don't touch this unless you really know what you're doing.

CONSTANT:
    Default: 120
    Also known as c_0 and c_1 in the Ryu Printf paper and SHIFT_CONST in
    float_to_string.h.
    The table value is shifted left by this amount, and the final value is
    shifted right by this amount. It effectively makes the table value a fixed
    point number with the decimal point at the bit that is CONSTANT bits from
    the right.
    Higher values increase accuracy, but also require higher MID_INT_WIDTH
    values to fit the result.

IDX_SIZE:
    Default: 16
    By increasing the MOD_SIZE slightly we can significantly decrease the number
    of table entries we need.
    We can divide the number of table entries by IDX_SIZE, and multiply MOD_SIZE
    by 2^IDX_SIZE, and the math still works out.
    This optimization isn't mentioned in the original Ryu Printf paper but it
    saves a lot of space.

MID_INT_WIDTH:
    Default: 192
    This is the size of integer that the tables use. It's called mid int because
    it's the integer used in the middle of the calculation. There are large ints
    used to calculate the mid int table values, then those are used to calculate
    the small int final values.
    This must be divisible by 64 since each table entry is an array of 64 bit
    integers.
    If this is too small, then the results will get cut off. It should be at
    least CONSTANT + IDX_SIZE + log_2(10^9) to be able to fit the table values.

MANT_WIDTH:
    The width of the widest float mantissa that this table will work for.
    This has a small effect on table size.

EXP_WIDTH:
    The width of the widest float exponent that this table will work for.
    This has a large effect on table size.
        Specifically, table size is proportional to the square of this number.
"""

BLOCK_SIZE = 9


# Values for double
# CONSTANT = 120
# IDX_SIZE = 16
# MANT_WIDTH = 52
# EXP_WIDTH = 11
# MID_INT_SIZE = 192

# Values for 128 bit float
CONSTANT = 120
IDX_SIZE = 128
MANT_WIDTH = 112
EXP_WIDTH = 15
MID_INT_SIZE = 256 + 64

MAX_EXP = 2 ** (EXP_WIDTH - 1)
POSITIVE_ARR_SIZE = MAX_EXP // IDX_SIZE
NEGATIVE_ARR_SIZE = (MAX_EXP // IDX_SIZE) + ((MANT_WIDTH + (IDX_SIZE - 1)) // IDX_SIZE)
MOD_SIZE = (10**BLOCK_SIZE) << (CONSTANT + (IDX_SIZE if IDX_SIZE > 1 else 0))


# floor(5^(-9i) * 2^(e + c_1 - 9i) + 1) % (10^9 * 2^c_1)
def get_table_positive(exponent, i):
    pow_of_two = 1 << (exponent + CONSTANT - (BLOCK_SIZE * i))
    pow_of_five = 5 ** (BLOCK_SIZE * i)
    result = (pow_of_two // pow_of_five) + 1
    return result % MOD_SIZE


# floor(10^(9*(-i)) * 2^(c_0 + (-e))) % (10^9 * 2^c_0)
def get_table_negative(exponent, i):
    result = 1
    pow_of_ten = 10 ** (BLOCK_SIZE * i)
    shift_amount = CONSTANT - exponent
    if shift_amount < 0:
        result = pow_of_ten >> (-shift_amount)
    else:
        result = pow_of_ten << (shift_amount)
    return result % MOD_SIZE


# Returns floor(log_10(2^e)); requires 0 <= e <= 42039.
def ceil_log10_pow2(e):
    return ((e * 0x13441350FBD) >> 42) + 1


def length_for_num(idx, index_size=IDX_SIZE):
    return (
        ceil_log10_pow2(idx * index_size) + ceil_log10_pow2(MANT_WIDTH) + BLOCK_SIZE - 1
    ) // BLOCK_SIZE


def get_64bit_window(num, index):
    return (num >> (index * 64)) % (2**64)


def mid_int_to_str(num):
    outstr = "    {"
    outstr += str(get_64bit_window(num, 0)) + "u"
    for i in range(1, MID_INT_SIZE // 64):
        outstr += ", " + str(get_64bit_window(num, i)) + "u"
    outstr += "},"
    return outstr


def print_positive_table_for_idx(idx):
    positive_blocks = length_for_num(idx)
    for i in range(positive_blocks):
        table_val = get_table_positive(idx * IDX_SIZE, i)
        # print(hex(table_val))
        print(mid_int_to_str(table_val))
    return positive_blocks


def print_negative_table_for_idx(idx):
    i = 0
    min_block = -1
    table_val = 0
    MIN_USEFUL_VAL = 2 ** (CONSTANT - (MANT_WIDTH + 2))
    # Iterate through the zero blocks
    while table_val < MIN_USEFUL_VAL:
        i += 1
        table_val = get_table_negative((idx) * IDX_SIZE, i)
    else:
        i -= 1

    min_block = i

    # Iterate until another zero block is found
    while table_val >= MIN_USEFUL_VAL:
        table_val = get_table_negative((idx) * IDX_SIZE, i + 1)
        if table_val >= MIN_USEFUL_VAL:
            print(mid_int_to_str(table_val))
            i += 1
    return i - min_block, min_block


positive_size_arr = [0] * (POSITIVE_ARR_SIZE + 1)

negative_size_arr = [0] * (NEGATIVE_ARR_SIZE + 1)
min_block_arr = [0] * (NEGATIVE_ARR_SIZE + 1)
acc = 0

if MOD_SIZE > (2**MID_INT_SIZE):
    print(
        "Mod size is too big for current MID_INT_SIZE by a factor of",
        MOD_SIZE // (2**MID_INT_SIZE),
    )
else:
    print("static const uint64_t POW10_SPLIT[][" + str(MID_INT_SIZE // 64) + "] = {")
    for idx in range(0, POSITIVE_ARR_SIZE + 1):
        num_size = print_positive_table_for_idx(idx)
        positive_size_arr[idx] = acc
        acc += num_size
    print("};")

    print(
        "static const uint32_t POW10_OFFSET_2[" + str(len(positive_size_arr)) + "] = {",
        str(positive_size_arr)[1:-2],
        "};",
    )

    print("static const uint64_t POW10_SPLIT_2[][" + str(MID_INT_SIZE // 64) + "] = {")
    for idx in range(0, NEGATIVE_ARR_SIZE):
        num_size, min_block = print_negative_table_for_idx(idx)
        acc += num_size
        negative_size_arr[idx + 1] = acc
        min_block_arr[idx] = min_block
    print("};")
    print(
        "static const uint32_t POW10_OFFSET_2[" + str(len(negative_size_arr)) + "] = {",
        str(negative_size_arr)[1:-2],
        "};",
    )
    print(
        "static const uint16_t MIN_BLOCK_2[" + str(len(min_block_arr)) + "] = {",
        str(min_block_arr)[1:-2],
        "};",
    )
