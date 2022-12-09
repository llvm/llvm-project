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
        if   i == '__assert': continue
        elif i == '__availability': continue
        elif i == '__bit_reference': continue
        elif i == '__bits': public = ['bits']
        elif i == '__bsd_locale_defaults.h': continue
        elif i == '__bsd_locale_fallbacks.h': continue
        elif i == '__config_site.in': continue
        elif i == '__config': continue
        elif i == '__debug': continue
        elif i == '__errc': continue
        elif i == '__hash_table': public = ['unordered_map', 'unordered_set']
        elif i == '__locale': public = ['locale']
        elif i == '__mbstate_t.h': continue
        elif i == '__mutex_base': continue
        elif i == '__node_handle': public = ['map', 'set', 'unordered_map', 'unordered_set']
        elif i == '__split_buffer': public = ['deque', 'vector']
        elif i == '__std_stream': public = ['iostream']
        elif i == '__threading_support': public = ['atomic', 'mutex', 'semaphore', 'thread']
        elif i == '__tree': public = ['map', 'set']
        elif i == '__undef_macros': continue
        elif i == '__verbose_abort': continue
        else: panic()

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
