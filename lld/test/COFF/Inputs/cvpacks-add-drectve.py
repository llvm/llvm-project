"""Copy a COFF YAML file, adding a .drectve section as its last section, so
that no section index changes. The section holds the given directives followed
by two NUL bytes of padding.

Usage: cvpacks-add-drectve.py <in.yaml> <out.yaml> <directives>
"""
import sys

src, dst, directives = sys.argv[1:4]
text = open(src).read()
data = (directives.encode() + b"\0\0").hex().upper()
section = (
    "  - Name:            .drectve\n"
    "    Characteristics: [ IMAGE_SCN_LNK_INFO, IMAGE_SCN_LNK_REMOVE ]\n"
    "    Alignment:       1\n"
    "    SectionData:     " + data + "\n"
)
marker = "\nsymbols:"
assert text.count(marker) == 1
text = text.replace(marker, "\n" + section.rstrip("\n") + marker)
open(dst, "w").write(text)
