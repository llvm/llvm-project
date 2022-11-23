#!/usr/bin/env python

import os, pathlib, sys

def generate(private, public):
    return f'{{ include: [ "{private}", "private", "<{public}>", "public" ] }}'


def panic(file):
    print(f'========== {__file__} error ==========', file=sys.stderr)
    print(f'\tFile \'{file}\' is a top-level detail header without a mapping', file=sys.stderr)
    sys.exit(1)


def generate_map(include):
    detail_files = []
    detail_directories = []
    c_headers = []

    for i in include.iterdir():
        if i.is_dir() and i.name.startswith('__'):
            detail_directories.append(f'{i.name}')
            continue

        if i.name.startswith('__'):
            detail_files.append(i.name)
            continue

        if i.name.endswith('.h'):
            c_headers.append(i.name)

    result = []
    for i in detail_directories:
        result.append(f'{generate(f"@<{i}/.*>", i[2:])},')

    for i in detail_files:
        public = []
        match i:
            case '__assert': continue
            case '__availability': continue
            case '__bit_reference': continue
            case '__bits': public = ['bits']
            case '__bsd_locale_defaults.h': continue
            case '__bsd_locale_fallbacks.h': continue
            case '__config_site.in': continue
            case '__config': continue
            case '__debug': continue
            case '__errc': continue
            case '__hash_table': public = ['unordered_map', 'unordered_set']
            case '__locale': public = ['locale']
            case '__mbstate_t.h': continue
            case '__mutex_base': continue
            case '__node_handle': public = ['map', 'set', 'unordered_map', 'unordered_set']
            case '__split_buffer': public = ['deque', 'vector']
            case '__std_stream': public = ['iostream']
            case '__threading_support': public = ['atomic', 'mutex', 'semaphore', 'thread']
            case '__tree': public = ['map', 'set']
            case '__undef_macros': continue
            case '__verbose_abort': continue
            case _: panic()

        for p in public:
            result.append(f'{generate(f"<{i}>", p)},')

    result.sort()
    return result

def main():
    monorepo_root = pathlib.Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    assert(monorepo_root.exists())
    include = pathlib.Path(os.path.join(monorepo_root, 'libcxx', 'include'))

    mapping = generate_map(include)
    data = '[\n  ' + '\n  '.join(mapping) + '\n]\n'
    with open(f'{include}/libcxx.imp', 'w') as f:
        f.write(data)


if __name__ == '__main__':
    main()
