# RUN: %{python} %s

# Verify that libcxx/include/CMakeLists.txt's list of header files
# is maintained in alphabetical order.

import os


if __name__ == "__main__":
    libcxx_test_libcxx_lint = os.path.dirname(os.path.abspath(__file__))
    libcxx = os.path.abspath(os.path.join(libcxx_test_libcxx_lint, "../../.."))
    cmakelists_name = os.path.join(libcxx, "include/CMakeLists.txt")
    assert os.path.isfile(cmakelists_name)

    with open(cmakelists_name, "r") as f:
        lines = f.readlines()

    assert lines[0] == "set(files\n"

    okay = True
    prevline = lines[1]
    for line in lines[2:]:
        if line == "  )\n":
            break
        if line < prevline:
            okay = False
            print("LINES OUT OF ORDER in libcxx/include/CMakeLists.txt!")
            print(prevline)
            print(line)
        prevline = line
    assert okay
