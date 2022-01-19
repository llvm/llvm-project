// RUN: %clang_cc1 -Werror -ivfsoverlay %S/Inputs/vfsoverlay-root-relative.yaml -I virtual -fsyntax-only %s

#include "virtual_header.h"
