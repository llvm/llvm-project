# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-win32 %s -o %t

# RUN: echo "A B C 100" > %t.call_graph
# RUN: not lld-link /dll /noentry  /subsystem:console %t /call-graph-ordering-file:%t.call_graph /out:/dev/null 2>&1 | FileCheck %s

# CHECK: {{.*}}.call_graph: parse error

# RUN: echo "A B C" > %t.call_graph
# RUN: not lld-link /dll /noentry  /subsystem:console %t /call-graph-ordering-file:%t.call_graph /out:/dev/null 2>&1 | FileCheck %s
