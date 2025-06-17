# Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
# amd/comgr/LICENSE.TXT in this repository for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from argparse import ArgumentParser
from hashlib import sha256
from os.path import join as join_path

if __name__ == "__main__":
    parser = ArgumentParser(description='Generate id by computing a hash of the generated headers')
    parser.add_argument("headers", nargs='+', help='List of headers to generate id from')
    # On Windows, we cannot list the realpath for every individual header since we hit cmd.exe's
    # maximum command line lenght. As a workaround, we pass the pwd and the headers separately.
    parser.add_argument("--parent-directory", help='Parent directory for the headers', required=True)
    parser.add_argument("--varname", help='Name of the variable to generate', required=True)
    parser.add_argument("--output", help='Name of the header to generate', required=True)

    args = parser.parse_args()
    args.headers.sort()
    
    hash = sha256()
    for header in args.headers:
        hash.update(open(join_path(args.parent_directory, header), 'rb').read())
    digest_uchar = hash.digest()
    digest_elts = ", ".join(map(str, digest_uchar))
    print(f"static const unsigned char {args.varname}[] = {{{digest_elts}, 0}};", file=open(args.output, 'w'))
