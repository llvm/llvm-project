# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir
# RUN: cd %t.dir

# RUN: not ld.lld --reproduce repro.tar abc -o t 2>&1 | FileCheck -DMSG=%errc_ENOENT %s
# CHECK: cannot open abc: [[MSG]]

# RUN: tar xOf repro.tar repro/response.txt | FileCheck --check-prefix=RSP %s
# RSP: abc
# RSP: -o t
