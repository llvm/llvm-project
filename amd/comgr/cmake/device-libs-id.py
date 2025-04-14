# Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
# amd/comgr/LICENSE.TXT in this repository for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from argparse import ArgumentParser
from hashlib import sha256
from functools import reduce

if __name__ == "__main__":
    parser = ArgumentParser(description='Generate id by computing a hash of the generated headers')
    parser.add_argument("headers", nargs='+', help='List of headers to generate id from')
    parser.add_argument("--varname", help='Name of the variable to generate', required=True)
    parser.add_argument("--output", help='Name of the header to generate', required=True)

    args = parser.parse_args()
    args.headers.sort()
    
    hash = sha256()
    for x in args.headers:
        hash.update(open(x, 'rb').read())
    digest_uchar = hash.digest()
    digest_elts = ", ".join(map(str, digest_uchar))
    print(f"static const unsigned char {args.varname}[] = {{{digest_elts}, 0}};", file=open(args.output, 'w'))
