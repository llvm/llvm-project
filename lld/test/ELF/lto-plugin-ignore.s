# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld %t -plugin-opt=/foo/bar -plugin-opt=-fresolution=zed \
# RUN:   -plugin-opt=-pass-through=-lgcc -plugin-opt=-function-sections \
# RUN:   -plugin-opt=-data-sections -plugin-opt=thinlto -o /dev/null

# RUN: not ld.lld %t -plugin-opt=-data-sectionxxx \
# RUN:   -plugin-opt=-function-sectionxxx 2>&1 | FileCheck %s
# CHECK: Unknown command line argument '-data-sectionxxx'
# CHECK: Unknown command line argument '-function-sectionxxx'
