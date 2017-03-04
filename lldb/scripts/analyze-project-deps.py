import os
import re

from use_lldb_suite import lldb_root

src_dir = os.path.join(lldb_root, "source")
inc_dir = os.path.join(lldb_root, "include")

src_map = {}

include_regex = re.compile('#include \"(lldb(.*/)+).*\"')

def scan_deps(this_dir, file):
    includes = set()
    with open(file) as f:
        for line in list(f):
            m = include_regex.match(line)
            if m is not None:
                relative = m.groups()[0].rstrip("/")
                if relative != this_dir:
                    includes.add(relative)
    return includes

def insert_or_add_mapping(base, deps):
    global src_map
    if len(deps) > 0:
        if base in src_map:
            existing_deps = src_map[base]
            existing_deps.update(deps)
        else:
            src_map[base] = deps

for (base, dirs, files) in os.walk(inc_dir):
    dir = os.path.basename(base)
    relative = os.path.relpath(base, inc_dir)
    inc_files = filter(lambda x : os.path.splitext(x)[1] in [".h"], files)
    deps = set()
    relative = relative.replace("\\", "/")
    for inc in inc_files:
        inc_path = os.path.join(base, inc)
        deps.update(scan_deps(relative, inc_path))
    insert_or_add_mapping(relative, deps)

for (base, dirs, files) in os.walk(src_dir):
    dir = os.path.basename(base)
    relative = os.path.relpath(base, src_dir)
    src_files = filter(lambda x : os.path.splitext(x)[1] in [".cpp", ".h", ".mm"], files)
    deps = set()
    norm_base_path = os.path.normpath(os.path.join("lldb", relative))
    norm_base_path = norm_base_path.replace("\\", "/")
    for src in src_files:
        src_path = os.path.join(base, src)
        deps.update(scan_deps(norm_base_path, src_path))
    insert_or_add_mapping(norm_base_path, deps)
    pass

items = list(src_map.iteritems())
items.sort(lambda A, B : cmp(A[0], B[0]))

for (path, deps) in items:
    print path + ":"
    sorted_deps = list(deps)
    sorted_deps.sort()
    for dep in sorted_deps:
        print "\t" + dep
pass