# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld %t -plugin-opt=/foo/bar -plugin-opt=-fresolution=zed \
# RUN:   -plugin-opt=-pass-through=-lgcc -o /dev/null
