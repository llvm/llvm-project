#!/usr/bin/env python3
"""
Generate FileCheck CHECK lines to match opt-remarks
YAML output.
"""
import argparse
import os
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('yaml_file',
        default=sys.stdin,
        type=argparse.FileType('r'),
        help='Input YAML file')
    parser.add_argument('--check-prefix',
        default='OPT-REM',
        help='FileCheck prefix. Default is "%(default)s".')
    parser.add_argument('-o',
        default=sys.stdout,
        type=argparse.FileType('w'),
        help='Output file. Default is stdout.')
    parser.add_argument('--ignore-whitespace-lines',
        action='store_true',
        help='Skip emitting CHECK directives for empty or whitespace lines')

    pargs = parser.parse_args()

    lines = get_check_lines(pargs)
    for l in lines:
        pargs.o.write(l)
        pargs.o.write('\n')
    pargs.o.flush()
    return 0

class Directive:
    def __init__(self, raw_line:str, prefix:str):
        self._prefix = prefix
        self._raw_line = raw_line

    def get_line(self) -> str:
        return f'{self._prefix}{self._raw_line}'


class CheckDirective(Directive):
    """
        Represents FileCheck's `CHECK:` directive.
    """
    def __init__(self, raw_line:str, prefix:str):
        assert isinstance(prefix, str) and len(prefix) > 0
        assert isinstance(raw_line, str) and len(raw_line) > 0
        super().__init__(raw_line, f'// {prefix}: ')

class CheckNextDirective(Directive):
    """
        Represents FileCheck's `CHECK-NEXT:` directive.
    """
    def __init__(self, raw_line:str, prefix:str):
        assert isinstance(prefix, str) and len(prefix) > 0
        assert isinstance(raw_line, str) and len(raw_line) > 0
        super().__init__(raw_line, f'// {prefix}-NEXT: ')

class CheckNotDirective(Directive):
    """
        Represents FileCheck's `CHECK-NOT:` directive.
    """
    def __init__(self, raw_line:str, prefix:str):
        assert isinstance(prefix, str) and len(prefix) > 0
        assert isinstance(raw_line, str) and len(raw_line) > 0
        super().__init__(raw_line, f'// {prefix}-NOT: ')

class CheckEmptyDirective(Directive):
    """
        Represents FileCheck's `CHECK-EMPTY:` directive.
    """
    def __init__(self, prefix:str):
        assert isinstance(prefix, str) and len(prefix) > 0
        super().__init__('', f'// {prefix}-EMPTY: ')

class EmptyLine(Directive):
    """
        This isn't a real FileCheck directive but its a
        useful mechanism for emitting empty lines in the
        final output
    """
    def __init__(self):
        super().__init__('', '')


__RE_DEBUG_LOC_LINE_START = re.compile(r"^(?P<head>\s*DebugLoc:\s+\{ File: ')(?P<file>.+)(?P<tail>',\s*$)$")
def filter_line(line:str) -> str:
    """
        Filter lines in opt-remark YAML output.
    """
    assert isinstance(line, str)
    
    # Strip paths in 'DebugLoc:' so that they are portable
    debug_loc_match = __RE_DEBUG_LOC_LINE_START.match(line)
    if debug_loc_match:
        file_path = debug_loc_match.group('file'
        )
        # Strip off the path prefix
        basename_file_path = os.path.basename(file_path)
        # Do replacement
        line = __RE_DEBUG_LOC_LINE_START.sub(
            r'\g<head>' + '{{.*}}' + basename_file_path + r'\g<tail>',
            line)

    return line

_RE_WHITESPACE_ONLY =  re.compile(r'^[ \t]+$')
def get_check_lines(pargs: argparse.Namespace) -> list[str]:
    """
        Generate a list of lines containing CHECK directives that
        try to match the provided YAML file.
    """
    directives = [ None ]
    lines = pargs.yaml_file.readlines()
    idx = [ 0 ] # Hack make lamba capture work
    max_line = len(lines) -1
    def read_line_and_inc():
        if idx[0] > max_line:
            raise Exception(
                f'Cannot read line {idx[0]+1} which is past last line '
                f'{max_line+1}')
        line_to_read = lines[idx[0]]
        idx[0] += 1
        return line_to_read.removesuffix('\n')
    def current_idx() -> int:
        return idx[0]

    NextDirectiveType = CheckDirective

    # Loop over lines
    while current_idx() <= max_line:
        current_line_idx = current_idx()
        current_line = read_line_and_inc()

        if current_line == '--- !Analysis' and current_line_idx != 0:
            # Add empty line between entries
            directives.append(EmptyLine())

        whitespace_only_match = _RE_WHITESPACE_ONLY.match(current_line)
        if whitespace_only_match:
            # Whitespace only lines
            if pargs.ignore_whitespace_lines:
                NextDirectiveType = CheckDirective
                continue
            
            # Craft a regex to match a line with just whitespace. This is
            # necessary because FileCheck doesn't allow most directives to be
            # empty.
            directive = NextDirectiveType('{{^[ \t]+$}}', prefix=pargs.check_prefix)
        elif len(current_line) == 0:
            # Empty line
            if pargs.ignore_whitespace_lines:
                NextDirectiveType = CheckDirective
                continue
            directive = CheckEmptyDirective(prefix=pargs.check_prefix)
        else:
            # Match the line.
            filtered_line = filter_line(current_line)
            directive = NextDirectiveType(filtered_line, prefix=pargs.check_prefix)

            NextDirectiveType = CheckNextDirective

        directives.append(directive)

    # Push CHECK-NOT to make sure there are no more entries
    directives.append(EmptyLine())
    directives.append(
        CheckNotDirective('--- !Analysis', prefix=pargs.check_prefix))

    # Loop over directives and gather the lines
    processed_lines = []
    for directive in directives:
        if directive is None:
            continue
        processed_lines.append(directive.get_line())
    return processed_lines 


if __name__ == '__main__':
    main()
