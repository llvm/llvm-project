#!/usr/bin/env python

# ULP error plot tool.
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import matplotlib.pyplot as plt
import sys
import re

# example usage:
# build/bin/ulp -e .0001 log 0.5 2.0 2345678 | math/tools/plot.py


def fhex(s):
    return float.fromhex(s)


def parse(f):
    xs = []
    gs = []
    ys = []
    es = []
    # Has to match the format used in ulp.c
    r = re.compile(r"[^ (]+\(([^ )]*)\) got ([^ ]+) want ([^ ]+) [^ ]+ ulp err ([^ ]+)")
    for line in f:
        m = r.match(line)
        if m:
            x = fhex(m.group(1))
            g = fhex(m.group(2))
            y = fhex(m.group(3))
            e = float(m.group(4))
            xs.append(x)
            gs.append(g)
            ys.append(y)
            es.append(e)
        elif line.startswith("PASS") or line.startswith("FAIL"):
            # Print the summary line
            print(line)
    return xs, gs, ys, es


def plot(xs, gs, ys, es):
    if len(xs) < 2:
        print("not enough samples")
        return
    a = min(xs)
    b = max(xs)
    fig, (ax0, ax1) = plt.subplots(nrows=2)
    es = np.abs(es)  # ignore the sign
    emax = max(es)
    ax0.text(a + (b - a) * 0.7, emax * 0.8, "%s\n%g" % (emax.hex(), emax))
    ax0.plot(xs, es, "r.")
    ax0.grid()
    ax1.plot(xs, ys, "r.", label="want")
    ax1.plot(xs, gs, "b.", label="got")
    ax1.grid()
    ax1.legend()
    plt.show()


xs, gs, ys, es = parse(sys.stdin)
plot(xs, gs, ys, es)
