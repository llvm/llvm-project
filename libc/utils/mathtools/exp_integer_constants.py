# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generate/check the exact constants for the integer-assisted binary64 exp.

No floating-point approximations are used to generate the constants.
Run with --check path/to/exp_integer_eval.h to verify a generated header.
Otherwise print the constant certificate as JSON.
"""
import argparse
from fractions import Fraction
import json
from math import factorial
from pathlib import Path
import re


def root_floor(value, degree):
    root = 1 << ((value.bit_length() + degree - 1) // degree)
    while True:
        next_root = ((degree - 1) * root + value // root ** (degree - 1)) // degree
        if next_root >= root:
            break
        root = next_root
    while root**degree > value:
        root -= 1
    while (root + 1) ** degree <= value:
        root += 1
    return root


def ln2_bounds(bits):
    # ln(2) = 2 * atanh(1/3). Bound the positive omitted tail geometrically.
    total = Fraction(0)
    for j in range(2 * bits):
        total += Fraction(2, (2 * j + 1) * 3 ** (2 * j + 1))
        tail = Fraction(9, 4 * (2 * j + 3) * 3 ** (2 * j + 3))
        lo = (total * (1 << bits)).__floor__()
        hi = ((total + tail) * (1 << bits)).__floor__()
        if lo == hi:
            return lo, lo + 1
    raise ArithmeticError("ln(2) floor not established")


def horner_bounds(coefficients, residual_bound):
    unit = 1 << 64
    upper = coefficients[6] + 1
    rows = []
    for degree in range(5, -1, -1):
        assert upper < 1 << 65
        product = (upper * residual_bound + unit - 1) // unit
        assert product < unit
        lower = coefficients[degree] - product
        upper = coefficients[degree] + int(degree > 2) + product
        assert lower > 0
        assert upper < (unit if degree >= 2 else 1 << 65)
        rows.append({"degree": degree, "lower": lower, "upper": upper})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    lnlo, lnhi = ln2_bounds(96)
    lower, upper = ln2_bounds(192)
    log2e = (1 << 255) // upper
    assert log2e == (1 << 255) // lower
    table = [root_floor(1 << (128 * 64 + i), 128) for i in range(128)]
    for i, value in enumerate(table):
        assert value**128 <= 1 << (128 * 64 + i) < (value + 1) ** 128
    inv = [(1 << 64) // factorial(n) for n in range(7)]
    residual_bound = (1 << 64) // 184
    assert residual_bound < 1 << 57
    assert table[0] == 1 << 64
    assert all(1 << 64 <= value < (1 << 65) - 1 for value in table)
    bounds = horner_bounds(inv, residual_bound)
    for exponent in range(0x3CA, 0x409):
        if exponent > 1003:
            assert 0 < 1067 - exponent < 64
        else:
            assert 1131 - exponent >= 128
    # The degree-six omitted Taylor tail is less than one Q64 unit.
    bound = Fraction(1, 184)
    tail = bound**7 / (factorial(7) * (1 - bound / 8))
    assert tail * (1 << 64) < 1
    result = {
        "ln2_lo": hex(lnlo),
        "ln2_hi": hex(lnhi),
        "log2e_63_floor": hex(log2e),
        "residual_bound": hex((1 << 64) // 184),
        "reciprocal_factorial_lo": [hex(v) for v in inv],
        "table_lo": [hex(v) for v in table],
        "taylor_tail_q64": str(tail * (1 << 64)),
        "horner_bounds": bounds,
    }
    if args.check:
        text = args.check.read_text()
        pairs = re.findall(
            r"\{\s*(0x[0-9a-f]+)ULL,\s*(0x[0-9a-f]+)ULL\s*\}",
            text.split("EXP2_TABLE[128]", 1)[1].split("};", 1)[0],
        )
        assert [int(lo, 16) + (int(hi, 16) << 64) for lo, hi in pairs] == table
        pattern = r"(?:big|UInt128)\((0x[0-9a-f]+)ULL\)\s*\|\s*\((?:big|UInt128)\((0x[0-9a-f]+)ULL\)\s*<<\s*64\)"
        ln = re.search(pattern, text.split("LN2_LO", 1)[1])
        assert ln and int(ln[1], 16) + (int(ln[2], 16) << 64) == lnlo
        coeffs = re.findall(
            pattern, text.split("INV_FACT_LO[7]", 1)[1].split("};", 1)[0]
        )
        assert [int(lo, 16) + (int(hi, 16) << 64) for lo, hi in coeffs] == inv
        actual = re.search(r"LOG2_E_63\s*=\s*(?:big|UInt128)\((0x[0-9a-f]+)ULL\)", text)
        assert actual and int(actual[1], 16) == log2e
        actual = re.search(r"R_BOUND\s*=\s*(?:big|UInt128)\((0x[0-9a-f]+)ULL\)", text)
        assert actual and int(actual[1], 16) == (1 << 64) // 184
        print(
            "Exact constants, tail, Horner capacity and reduction shift certificates passed"
        )
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
