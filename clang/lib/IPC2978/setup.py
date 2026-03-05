#!/usr/bin/env python3
import os
import shutil

# The directory where this script resides (lib/clang/IPC2978)
source_dir = os.path.abspath(os.path.dirname(__file__))
# Assumed that IPC2978 repo is present in parallel with llvm repo.
copy_from = os.path.abspath(source_dir + ("../" * 5) + "ipc2978api/") + os.sep
# The directory for the header-files
include_dir = (
    os.path.abspath(os.path.join(source_dir, "../../include/clang/IPC2978")) + os.sep
)
# The directory for the unit-tests.
ipc_test_source_file = os.path.abspath(
    os.path.join(source_dir, "../../unittests/IPC2978/IPC2978Test.cpp")
)

shutil.copytree(copy_from + "include", include_dir, dirs_exist_ok=True)
shutil.copytree(copy_from + "src", source_dir, dirs_exist_ok=True)
# We'll process files in both include and source directories
roots = [source_dir, include_dir]

# Gather all header filenames in the include directory (top-level only)
include_files = [f for f in os.listdir(include_dir) if f.endswith(".hpp")]

files = []

# Iterate through the source and include directories
for root in roots:
    # Skipping the CMakeLists.txt and .py files
    files.extend(
        [
            os.path.join(root, x)
            for x in os.listdir(root)
            if not os.path.join(root, x).endswith(".txt")
            and not os.path.join(root, x).endswith(".py")
        ]
    )

for file in files:
    out_lines = []
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # Examine each line for an include directive
        for line in lines:
            if line.startswith('#include "'):
                out_lines.append(
                    line.replace('#include "', '#include "clang/IPC2978/', 1)
                )
            else:
                out_lines.append(line)
    with open(file, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.writelines(line)

shutil.copy(copy_from + "/tests/ClangTest.cpp", ipc_test_source_file)

# Modifying the copied ClangTest.cpp file
out_lines = []
with open(ipc_test_source_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    # Examine each line for an include directive
    for line in lines:
        if line.startswith("// #define IS_THIS_CLANG_REPO"):
            out_lines.append("#define IS_THIS_CLANG_REPO")
        else:
            out_lines.append(line)
with open(ipc_test_source_file, "w", encoding="utf-8") as f:
    for line in out_lines:
        f.writelines(line)
