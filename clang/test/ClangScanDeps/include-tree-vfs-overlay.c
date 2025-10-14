// REQUIRES: ondisk_cas

// This is a reduced test for https://github.com/swiftlang/llvm-project/issues/11616

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/vfs-overlay.yaml.template > %t/vfs-overlay.yaml

// RUN: %clang -I%t -fdepscan=inline -fdepscan-include-tree -Xclang -fcas-path -Xclang %t/cas -Xclang -ivfsoverlay -Xclang %t/vfs-overlay.yaml -c %t/test.c

//--- test.c
#include "header.h"
#include "header1.h"

//--- header.h

//--- header2.h

//--- header3.h

//--- vfs-overlay.yaml.template
version: 0
case-sensitive: false
roots:
  - name: "DIR"
    type: directory
    contents:
      - name: header.h
        type: file
        external-contents: "DIR/header.h"
      - name: header1.h
        type: file
        external-contents: "DIR/header2.h"
      - name: header2.h
        type: file
        external-contents: "DIR/header3.h"
