from subprocess import check_output
import argparse
import re


def gen_llvm_c_export(output, libs, nm, dlltool, dllname):
    """Generate exporting library file for clang-cpp.dll.

    Run `nm` for each lib in `libs`, and output an exporting library
    file referring `dllname` to `output` using `dlltool`.
    """
    r = re.compile(r"^[0-9a-fA-F]+ ([A-Z]) (__?Z[0-9A-Za-z_]+)$")

    with open(output + ".def", "w+t") as output_f:
        output_f.write("LIBRARY " + dllname + "\n")
        output_f.write("EXPORTS\n")
        # For each lib get the LLVM* functions it exports.
        for lib in libs:
            # Capture nm output directly to avoid temp file issues on Windows
            # during parallel builds.
            nm_output = check_output([nm, "-Ugp", lib], text=True)

            # Get the matching lines.
            for line in nm_output.splitlines():
                m = r.match(line)
                if not m:
                    continue
                t = str(m.group(1))
                if t == "T":
                    output_f.write("  " + m.group(2) + "\n")
                elif t in "BCDGIRS":
                    output_f.write("  " + m.group(2) + " DATA\n")

    check_output([dlltool, "-d", output + ".def", "-D", dllname, "-l", output])


def main():
    parser = argparse.ArgumentParser("gen-msvc-exports")

    parser.add_argument(
        "-i",
        "--libsfile",
        help="file with list of libs, semicolon separated",
        action="store",
        default=None,
    )
    parser.add_argument("-o", "--output", help="output filename")
    parser.add_argument("-d", "--dllname", help="DLL name will be referred to")
    parser.add_argument(
        "--nm", help="path to the llvm-nm executable", default="llvm-nm"
    )
    parser.add_argument(
        "--dlltool", help="path to the llvm-dlltool executable", default="llvm-dlltool"
    )
    parser.add_argument(
        "libs",
        metavar="LIBS",
        nargs="*",
        help="list of libraries to generate export from",
    )

    ns = parser.parse_args()

    libs = ns.libs

    # Add if we where given a libsfile add it to the libs.
    if ns.libsfile:
        with open(ns.libsfile) as f:
            libs.extend(f.read().replace(";", "\n").splitlines())

    gen_llvm_c_export(ns.output, libs, ns.nm, ns.dlltool, ns.dllname)


if __name__ == "__main__":
    main()
