#!/usr/bin/env python3
"""Update Options.td for the flags changes in https://reviews.llvm.org/Dxyz

This script translates Options.td from using Flags to control option visibility
to using Vis instead. It is meant to be idempotent and usable to help update
downstream forks if they have their own changes to Options.td.

Usage:
```sh
% update_options_td_flags.py path/to/Options.td > Options.td.new
% mv Options.td.new path/to/Options.td
```

This script will be removed after the next LLVM release.
"""

import argparse
import re
import shutil
import sys
import tempfile

def rewrite_option_flags(input_file, output_file):
    for src_line in input_file:
        for dst_line in process_line(src_line):
            output_file.write(dst_line)

def process_line(line):
    # We only deal with one thing per line. If multiple things can be
    # on the same line (like NegFlag and PosFlag), please preprocess
    # that first.
    m = re.search(r'((NegFlag|PosFlag)<[A-Za-z]+, |BothFlags<)'
                  r'\[([A-Za-z0-9, ]+)\](, \[ClangOption\]|(?=>))', line)
    if m:
        return process_boolflags(m.group(3), line[:m.end(1)], line[m.end():])
    m = re.search(r'\bFlags<\[([A-Za-z0-9, ]*)\]>', line)
    if m:
        return process_flags(m.group(1), line[:m.start()], line[m.end():])
    m = re.search(r'let Flags = \[([A-Za-z0-9, ]*)\]', line)
    if m:
        return process_letflags(m.group(1), line[:m.start(1)], line[m.end():])

    return [line]

def process_boolflags(flag_group, prefix, suffix):
    flags = [f.strip() for f in flag_group.split(',')] if flag_group else []
    if not flags:
        return f'{prefix}[], [ClangOption]{suffix}'

    flags_to_keep, vis_mods = translate_flags(flags)
    flag_text = f'[{", ".join(flags_to_keep)}]'
    vis_text = f'[{", ".join(vis_mods)}]'
    new_text = ', '.join([flag_text, vis_text])

    if prefix.startswith('Both'):
        indent = ' ' * len(prefix)
    else:
        indent = ' ' * (len(prefix) - len(prefix.lstrip()) + len('XyzFlag<'))

    return get_edited_lines(prefix, new_text, suffix, indent=indent)

def process_flags(flag_group, prefix, suffix):
    flags = [f.strip() for f in flag_group.split(',')]

    flags_to_keep, vis_mods = translate_flags(flags)

    flag_text = ''
    vis_text = ''
    if flags_to_keep:
        flag_text = f'Flags<[{", ".join(flags_to_keep)}]>'
        if vis_mods:
            flag_text += ', '
    if vis_mods:
        vis_text = f'Visibility<[{", ".join(vis_mods)}]>'

    return get_edited_lines(prefix, flag_text, vis_text, suffix)

def process_letflags(flag_group, prefix, suffix):
    is_end_comment = prefix.startswith('} //')
    if not is_end_comment and not prefix.startswith('let'):
        raise AssertionError(f'Unusual let block: {prefix}')

    flags = [f.strip() for f in flag_group.split(',')]

    flags_to_keep, vis_mods = translate_flags(flags)

    lines = []
    if flags_to_keep:
        lines += [f'let Flags = [{", ".join(flags_to_keep)}]']
    if vis_mods:
        lines += [f'let Visibility = [{", ".join(vis_mods)}]']

    if is_end_comment:
        lines = list(reversed([f'}} // {l}\n' for l in lines]))
    else:
        lines = [f'{l} in {{\n' for l in lines]
    return lines

def get_edited_lines(prefix, *fragments, indent='  '):
    out_lines = []
    current = prefix
    for fragment in fragments:
        if fragment and len(current) + len(fragment) > 80:
            # Make a minimal attempt at reasonable line lengths
            if fragment.startswith(',') or fragment.startswith(';'):
                # Avoid wrapping the , or ; to the new line
                current += fragment[0]
                fragment = fragment[1:].lstrip()
            out_lines += [current.rstrip() + '\n']
            current = max(' ' * (len(current) - len(current.lstrip())), indent)
        current += fragment

    if current.strip():
        out_lines += [current]
    return out_lines

def translate_flags(flags):
    driver_flags = [
        'HelpHidden',
        'RenderAsInput',
        'RenderJoined',
        'RenderSeparate',
    ]
    custom_flags = [
        'NoXarchOption',
        'LinkerInput',
        'NoArgumentUnused',
        'Unsupported',
        'LinkOption',
        'Ignored',
        'TargetSpecific',
    ]
    flag_to_vis = {
        'CoreOption': ['ClangOption', 'CLOption', 'DXCOption'],
        'CLOption': ['CLOption'],
        'CC1Option': ['ClangOption', 'CC1Option'],
        'CC1AsOption': ['ClangOption', 'CC1AsOption'],
        'FlangOption': ['ClangOption', 'FlangOption'],
        'FC1Option': ['ClangOption', 'FC1Option'],
        'DXCOption': ['DXCOption'],
        'CLDXCOption': ['CLOption', 'DXCOption'],
    }
    new_flags = []
    vis_mods = []
    has_no_driver = False
    has_flang_only = False
    for flag in flags:
        if flag in driver_flags or flag in custom_flags:
            new_flags += [flag]
        elif flag in flag_to_vis:
            vis_mods += flag_to_vis[flag]
        elif flag == 'NoDriverOption':
            has_no_driver = True
        elif flag == 'FlangOnlyOption':
            has_flang_only = True
        else:
            raise AssertionError(f'Unknown flag: {flag}')

    new_vis_mods = []
    for vis in vis_mods:
        if vis in new_vis_mods:
            continue
        if has_no_driver and vis == 'ClangOption':
            continue
        if has_flang_only and vis == 'ClangOption':
            continue
        new_vis_mods += [vis]

    return new_flags, new_vis_mods

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', nargs='?', default='-',
                        type=argparse.FileType('r', encoding='UTF-8'))
    parser.add_argument('-o', dest='dst', default='-',
                        type=argparse.FileType('w', encoding='UTF-8'))

    args = parser.parse_args()
    rewrite_option_flags(args.src, args.dst)

if __name__ == '__main__':
    main()
