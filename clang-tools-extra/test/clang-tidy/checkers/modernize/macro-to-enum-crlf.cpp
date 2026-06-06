// RUN: %python -c "from pathlib import Path; Path(r'%/t.cpp').write_bytes(b'#define RED 1\r\n#define GREEN 2\r\n')"
// RUN: clang-tidy %t.cpp -fix --checks='-*,modernize-macro-to-enum' --config={} -- --std=c++14 > %t.out 2>&1
// RUN: %python -c "from pathlib import Path; data = Path(r'%/t.cpp').read_bytes(); expected = b'enum {\r\nRED = 1,\r\nGREEN = 2\r\n};\r\n'; assert data == expected, data"
