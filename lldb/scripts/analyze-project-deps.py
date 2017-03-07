import argparse
import os
import re

from use_lldb_suite import lldb_root

parser = argparse.ArgumentParser(
    description='Analyze LLDB project #include dependencies.')
parser.add_argument('--show-counts', default=False, action='store_true', 
    help='When true, show the number of dependencies from each subproject')
args = parser.parse_args()

src_dir = os.path.join(lldb_root, "source")
inc_dir = os.path.join(lldb_root, "include")

src_map = {}

include_regex = re.compile('#include \"((lldb|Plugins|clang)(.*/)+).*\"')

def normalize_host(str):
    if str.startswith("lldb/Host"):
        return "lldb/Host"
    return str

def scan_deps(this_dir, file):
    global src_map
    deps = {}
    this_dir = normalize_host(this_dir)
    if this_dir in src_map:
        deps = src_map[this_dir]

    with open(file) as f:
        for line in list(f):
            m = include_regex.match(line)
            if m is None:
                continue
            relative = m.groups()[0].rstrip("/")
            if relative == this_dir:
                continue
            relative = normalize_host(relative)
            if relative in deps:
                deps[relative] += 1
            else:
                deps[relative] = 1
    if this_dir not in src_map and len(deps) > 0:
        src_map[this_dir] = deps

for (base, dirs, files) in os.walk(inc_dir):
    dir = os.path.basename(base)
    relative = os.path.relpath(base, inc_dir)
    inc_files = filter(lambda x : os.path.splitext(x)[1] in [".h"], files)
    relative = relative.replace("\\", "/")
    for inc in inc_files:
        inc_path = os.path.join(base, inc)
        scan_deps(relative, inc_path)

for (base, dirs, files) in os.walk(src_dir):
    dir = os.path.basename(base)
    relative = os.path.relpath(base, src_dir)
    src_files = filter(lambda x : os.path.splitext(x)[1] in [".cpp", ".h", ".mm"], files)
    norm_base_path = os.path.normpath(os.path.join("lldb", relative))
    norm_base_path = norm_base_path.replace("\\", "/")
    for src in src_files:
        src_path = os.path.join(base, src)
        scan_deps(norm_base_path, src_path)
    pass

items = list(src_map.iteritems())
items.sort(lambda A, B : cmp(A[0], B[0]))

for (path, deps) in items:
    print path + ":"
    sorted_deps = list(deps.iteritems())
    if args.show_counts:
        sorted_deps.sort(lambda A, B: cmp(A[1], B[1]))
        for dep in sorted_deps:
            print "\t{} [{}]".format(dep[0], dep[1])
    else:
        sorted_deps.sort(lambda A, B: cmp(A[0], B[0]))
        for dep in sorted_deps:
            print "\t{}".format(dep[0])
pass