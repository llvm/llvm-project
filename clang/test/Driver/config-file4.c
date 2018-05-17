// RUN: %clang --config %S/Inputs/empty.cfg -Wall -Wextra -Wformat -Wstrict-aliasing -Wshadow -Wpacked -Winline -Wimplicit-function-declaration -c %s -O2 -o /dev/null -v 2>&1 | FileCheck %s -check-prefix PR37196
// PR37196: Configuration file: {{.*}}/empty.cfg
