import sys

path, offset, old, new = (sys.argv[1], int(sys.argv[2]),
                          int(sys.argv[3], 0), int(sys.argv[4], 0))
with open(path, "r+b") as f:
    f.seek(offset)
    actual = f.read(1)[0]
    assert actual == old, f"{path}:{offset}: found {actual:#x}, want {old:#x}"
    f.seek(offset)
    f.write(bytes([new]))
