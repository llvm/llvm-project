import sys
import re

"""
 Pipe output from clang's -verify into this script to have the test case updated to expect the actual diagnostic output.
 When inserting new expected-* checks it will place them on the line before the location of the diagnostic, with an @+1,
 or @+N for some N if there are multiple diagnostics emitted on the same line. If the current checks are using @-N for
 this line, the new check will follow that convention also.
 Existing checks will be left untouched as much as possible, including their location and whitespace content, to minimize
 diffs. If inaccurate their count will be updated, or the check removed entirely.

 Missing features:
  - custom prefix support (-verify=my-prefix)
  - multiple prefixes on the same line (-verify=my-prefix,my-other-prefix)
  - multiple prefixes on separate RUN lines (RUN: -verify=my-prefix\nRUN: -verify my-other-prefix)
  - regexes with expected-*-re: existing ones will be left untouched if accurate, but the script will abort if there are any
    diagnostic mismatches on the same line.
  - multiple checks targeting the same line are supported, but a line may only contain one check
  - if multiple checks targeting the same line are failing the script is not guaranteed to produce a minimal diff

Example usage:
  build/bin/llvm-lit clang/test/Sema/ --no-progress-bar -v | python3 update-verify-tests.py
"""


class KnownException(Exception):
    pass


def parse_error_category(s):
    parts = s.split("diagnostics")
    diag_category = parts[0]
    category_parts = parts[0].strip().strip("'").split("-")
    expected = category_parts[0]
    if expected != "expected":
        raise Exception(
            f"expected 'expected', but found '{expected}'. Custom verify prefixes are not supported."
        )
    diag_category = category_parts[1]
    if "seen but not expected" in parts[1]:
        seen = True
    elif "expected but not seen" in parts[1]:
        seen = False
    else:
        raise KnownException(f"unexpected category '{parts[1]}'")
    return (diag_category, seen)


diag_error_re = re.compile(r"File (\S+) Line (\d+): (.+)")
diag_error_re2 = re.compile(r"File \S+ Line \d+ \(directive at (\S+):(\d+)\): (.+)")


def parse_diag_error(s):
    m = diag_error_re2.match(s)
    if not m:
        m = diag_error_re.match(s)
    if not m:
        return None
    return (m.group(1), int(m.group(2)), m.group(3))


class Line:
    def __init__(self, content, line_n):
        self.content = content
        self.diag = None
        self.line_n = line_n
        self.related_diags = []
        self.targeting_diags = []

    def update_line_n(self, n):
        if self.diag and not self.diag.line_is_absolute:
            self.diag.orig_target_line_n += n - self.line_n
        self.line_n = n
        for diag in self.targeting_diags:
            if diag.line_is_absolute:
                diag.orig_target_line_n = n
            else:
                diag.orig_target_line_n = n - diag.line.line_n
        for diag in self.related_diags:
            if not diag.line_is_absolute:
                pass

    def render(self):
        if not self.diag:
            return self.content
        assert "{{DIAG}}" in self.content
        res = self.content.replace("{{DIAG}}", self.diag.render())
        if not res.strip():
            return ""
        return res


class Diag:
    def __init__(
        self,
        diag_content,
        category,
        targeted_line_n,
        line_is_absolute,
        count,
        line,
        is_re,
        whitespace_strings,
    ):
        self.diag_content = diag_content
        self.category = category
        self.orig_target_line_n = targeted_line_n
        self.line_is_absolute = line_is_absolute
        self.count = count
        self.line = line
        self.target = None
        self.is_re = is_re
        self.absolute_target()
        self.whitespace_strings = whitespace_strings

    def add(self):
        if targeted_line > 0:
            targeted_line += 1
        elif targeted_line < 0:
            targeted_line -= 1

    def absolute_target(self):
        if self.line_is_absolute:
            res = self.orig_target_line_n
        else:
            res = self.line.line_n + self.orig_target_line_n
        if self.target:
            assert self.line.line_n == res
        return res

    def relative_target(self):
        return self.absolute_target() - self.line.line_n

    def render(self):
        assert self.count >= 0
        if self.count == 0:
            return ""
        line_location_s = ""
        if self.relative_target() != 0:
            if self.line_is_absolute:
                line_location_s = f"@{self.absolute_target()}"
            elif self.relative_target() > 0:
                line_location_s = f"@+{self.relative_target()}"
            else:
                line_location_s = (
                    f"@{self.relative_target()}"  # the minus sign is implicit
                )
        count_s = "" if self.count == 1 else f"{self.count}"
        re_s = "-re" if self.is_re else ""
        if self.whitespace_strings:
            whitespace1_s = self.whitespace_strings[0]
            whitespace2_s = self.whitespace_strings[1]
            whitespace3_s = self.whitespace_strings[2]
            whitespace4_s = self.whitespace_strings[3]
        else:
            whitespace1_s = " "
            whitespace2_s = ""
            whitespace3_s = ""
            whitespace4_s = ""
        if count_s and not whitespace3_s:
            whitespace3_s = " "
        return f"//{whitespace1_s}expected-{self.category}{re_s}{whitespace2_s}{line_location_s}{whitespace3_s}{count_s}{whitespace4_s}{{{{{self.diag_content}}}}}"


expected_diag_re = re.compile(
    r"//(\s*)expected-(note|warning|error)(-re)?(\s*)(@[+-]?\d+)?(\s*)(\d+)?(\s*)\{\{(.*)\}\}"
)


def parse_diag(line, filename, lines):
    s = line.content
    ms = expected_diag_re.findall(s)
    if not ms:
        return None
    if len(ms) > 1:
        print(
            f"multiple diags on line {filename}:{line.line_n}. Aborting due to missing implementation."
        )
        sys.exit(1)
    [
        whitespace1_s,
        category_s,
        re_s,
        whitespace2_s,
        target_line_s,
        whitespace3_s,
        count_s,
        whitespace4_s,
        diag_s,
    ] = ms[0]
    if not target_line_s:
        target_line_n = 0
        is_absolute = False
    elif target_line_s.startswith("@+"):
        target_line_n = int(target_line_s[2:])
        is_absolute = False
    elif target_line_s.startswith("@-"):
        target_line_n = int(target_line_s[1:])
        is_absolute = False
    else:
        target_line_n = int(target_line_s[1:])
        is_absolute = True
    count = int(count_s) if count_s else 1
    line.content = expected_diag_re.sub("{{DIAG}}", s)

    return Diag(
        diag_s,
        category_s,
        target_line_n,
        is_absolute,
        count,
        line,
        bool(re_s),
        [whitespace1_s, whitespace2_s, whitespace3_s, whitespace4_s],
    )


def link_line_diags(lines, diag):
    line_n = diag.line.line_n
    target_line_n = diag.absolute_target()
    step = 1 if target_line_n < line_n else -1
    for i in range(target_line_n, line_n, step):
        lines[i - 1].related_diags.append(diag)


def add_line(new_line, lines):
    lines.insert(new_line.line_n - 1, new_line)
    for i in range(new_line.line_n, len(lines)):
        line = lines[i]
        assert line.line_n == i
        line.update_line_n(i + 1)
    assert all(line.line_n == i + 1 for i, line in enumerate(lines))


indent_re = re.compile(r"\s*")


def get_indent(s):
    return indent_re.match(s).group(0)


def add_diag(line_n, diag_s, diag_category, lines):
    target = lines[line_n - 1]
    for other in target.targeting_diags:
        if other.is_re:
            raise KnownException(
                "mismatching diag on line with regex matcher. Skipping due to missing implementation"
            )
    reverse = (
        True
        if [other for other in target.targeting_diags if other.relative_target() < 0]
        else False
    )

    targeting = [
        other for other in target.targeting_diags if not other.line_is_absolute
    ]
    targeting.sort(reverse=reverse, key=lambda d: d.relative_target())
    prev_offset = 0
    prev_line = target
    direction = -1 if reverse else 1
    for d in targeting:
        if d.relative_target() != prev_offset + direction:
            break
        prev_offset = d.relative_target()
        prev_line = d.line
    total_offset = prev_offset - 1 if reverse else prev_offset + 1
    if reverse:
        new_line_n = prev_line.line_n + 1
    else:
        new_line_n = prev_line.line_n
    assert new_line_n == line_n + (not reverse) - total_offset

    new_line = Line(get_indent(prev_line.content) + "{{DIAG}}\n", new_line_n)
    new_line.related_diags = list(prev_line.related_diags)
    add_line(new_line, lines)

    new_diag = Diag(
        diag_s, diag_category, total_offset, False, 1, new_line, False, None
    )
    new_line.diag = new_diag
    new_diag.target_line = target
    assert type(new_diag) != str
    target.targeting_diags.append(new_diag)
    link_line_diags(lines, new_diag)


updated_test_files = set()


def update_test_file(filename, diag_errors):
    print(f"updating test file {filename}")
    if filename in updated_test_files:
        print(
            f"{filename} already updated, but got new output - expect incorrect results"
        )
    else:
        updated_test_files.add(filename)
    with open(filename, "r") as f:
        lines = [Line(line, i + 1) for i, line in enumerate(f.readlines())]
    for line in lines:
        diag = parse_diag(line, filename, lines)
        if diag:
            line.diag = diag
            diag.target_line = lines[diag.absolute_target() - 1]
            link_line_diags(lines, diag)
            lines[diag.absolute_target() - 1].targeting_diags.append(diag)

    for line_n, diag_s, diag_category, seen in diag_errors:
        if seen:
            continue
        # this is a diagnostic expected but not seen
        assert lines[line_n - 1].diag
        if diag_s != lines[line_n - 1].diag.diag_content:
            raise KnownException(
                f"{filename}:{line_n} - found diag {lines[line_n - 1].diag.diag_content} but expected {diag_s}"
            )
        if diag_category != lines[line_n - 1].diag.category:
            raise KnownException(
                f"{filename}:{line_n} - found {lines[line_n - 1].diag.category} diag but expected {diag_category}"
            )
        lines[line_n - 1].diag.count -= 1
    diag_errors_left = []
    diag_errors.sort(reverse=True, key=lambda t: t[0])
    for line_n, diag_s, diag_category, seen in diag_errors:
        if not seen:
            continue
        target = lines[line_n - 1]
        other_diags = [
            d
            for d in target.targeting_diags
            if d.diag_content == diag_s and d.category == diag_category
        ]
        other_diag = other_diags[0] if other_diags else None
        if other_diag:
            other_diag.count += 1
        else:
            diag_errors_left.append((line_n, diag_s, diag_category))
    for line_n, diag_s, diag_category in diag_errors_left:
        add_diag(line_n, diag_s, diag_category, lines)
    with open(filename, "w") as f:
        for line in lines:
            f.write(line.render())


def update_test_files(errors):
    errors_by_file = {}
    for (filename, line, diag_s), (diag_category, seen) in errors:
        if filename not in errors_by_file:
            errors_by_file[filename] = []
        errors_by_file[filename].append((line, diag_s, diag_category, seen))
    for filename, diag_errors in errors_by_file.items():
        try:
            update_test_file(filename, diag_errors)
        except KnownException as e:
            print(f"{filename} - ERROR: {e}")
            print("continuing...")


curr = []
curr_category = None
curr_run_line = None
lines_since_run = []
for line in sys.stdin.readlines():
    lines_since_run.append(line)
    try:
        if line.startswith("RUN:"):
            if curr:
                update_test_files(curr)
                curr = []
                lines_since_run = [line]
                curr_run_line = line
            else:
                for line in lines_since_run:
                    print(line, end="")
                    print("====================")
                print("no mismatching diagnostics found since last RUN line")
            continue
        if line.startswith("error: "):
            if "no expected directives found" in line:
                print(
                    f"no expected directives found for RUN line '{curr_run_line.strip()}'. Add 'expected-no-diagnostics' manually if this is intended."
                )
                continue
            curr_category = parse_error_category(line[len("error: ") :])
            continue

        diag_error = parse_diag_error(line.strip())
        if diag_error:
            curr.append((diag_error, curr_category))
    except Exception as e:
        for line in lines_since_run:
            print(line, end="")
            print("====================")
            print(e)
        sys.exit(1)
if curr:
    update_test_files(curr)
    print("done!")
else:
    for line in lines_since_run:
        print(line, end="")
        print("====================")
    print("no mismatching diagnostics found")
