// RUN: %python -c "open(r'%t.h','wb').write(open(r'%S/Inputs/comment_save_crlf.h','rb').read().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))"
// RUN: %python -c "import sys; sys.exit(b'\r\n' not in open(r'%t.h','rb').read())"
// RUN: %clang_cc1 -E -C -o %t.i %t.h
// RUN: %python -c "import sys; sys.exit(b'\r\r\n' in open(r'%t.i','rb').read())"
