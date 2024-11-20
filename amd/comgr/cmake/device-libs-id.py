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
    digest_char = [e if e < 128 else e-256 for e in digest_uchar]
    digest_elts = ", ".join(map(str, digest_char))
    print(f"static const char {args.varname}[] = {{{digest_elts}, 0}};", file=open(args.output, 'w'))
